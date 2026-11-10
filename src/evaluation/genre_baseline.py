from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.labels import CLASSES_FINAIS
from src.evaluation.evaluate_model import (
    calcular_metricas,
    salvar_matriz_confusao,
)


CLASS_TO_IDX = {
    classe: idx
    for idx, classe in enumerate(
        CLASSES_FINAIS
    )
}

REQUIRED_COLUMNS = {
    "Painting_ID",
    "Macro_Estilo",
    "Genero",
}


def carregar_metadata(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    faltantes = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if faltantes:
        raise ValueError(
            f"{path} não contém as colunas "
            f"necessárias: {sorted(faltantes)}"
        )

    if df["Painting_ID"].duplicated().any():
        raise ValueError(
            f"{path} contém Painting_ID duplicado."
        )

    return df


def aprender_mapa_genero(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Para cada gênero, aprende qual macroclasse
    é mais frequente no conjunto de treino.

    Em caso de empate, usa a ordem oficial de
    CLASSES_FINAIS apenas como desempate
    determinístico.
    """

    contagens = (
        train_df.groupby(
            [
                "Genero",
                "Macro_Estilo",
            ],
            observed=True,
        )
        .size()
        .reset_index(
            name="n_class"
        )
    )

    totais_genero = (
        contagens.groupby(
            "Genero"
        )["n_class"]
        .sum()
        .rename("n_genre_train")
    )

    contagens = contagens.merge(
        totais_genero,
        on="Genero",
        how="left",
    )

    contagens[
        "class_share_train"
    ] = (
        contagens["n_class"]
        / contagens["n_genre_train"]
    )

    contagens["class_idx"] = (
        contagens[
            "Macro_Estilo"
        ].map(CLASS_TO_IDX)
    )

    contagens = (
        contagens.sort_values(
            [
                "Genero",
                "n_class",
                "class_idx",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    max_por_genero = (
        contagens.groupby(
            "Genero"
        )["n_class"]
        .transform("max")
    )

    contagens["is_max"] = (
        contagens["n_class"]
        == max_por_genero
    )

    empates = (
        contagens[
            contagens["is_max"]
        ]
        .groupby("Genero")
        .size()
        .rename("n_classes_tied")
    )

    mapa = (
        contagens
        .drop_duplicates(
            subset=["Genero"],
            keep="first",
        )
        .merge(
            empates,
            on="Genero",
            how="left",
        )
    )

    mapa = mapa.rename(
        columns={
            "Macro_Estilo":
                "predicted_class",
            "class_idx":
                "predicted_idx",
        }
    )

    return mapa[
        [
            "Genero",
            "predicted_class",
            "predicted_idx",
            "n_class",
            "n_genre_train",
            "class_share_train",
            "n_classes_tied",
        ]
    ]


def aplicar_baseline(
    test_df: pd.DataFrame,
    mapa: pd.DataFrame,
) -> pd.DataFrame:

    generos_train = set(
        mapa["Genero"]
    )

    generos_test = set(
        test_df["Genero"]
    )

    nao_vistos = (
        generos_test
        - generos_train
    )

    if nao_vistos:
        raise ValueError(
            "O teste contém gêneros não observados "
            f"no treino: {sorted(nao_vistos)}"
        )

    resultado = test_df[
        [
            "Painting_ID",
            "Macro_Estilo",
            "Genero",
        ]
    ].merge(
        mapa[
            [
                "Genero",
                "predicted_class",
                "predicted_idx",
            ]
        ],
        on="Genero",
        how="left",
        validate="many_to_one",
    )

    resultado["true_idx"] = (
        resultado[
            "Macro_Estilo"
        ].map(CLASS_TO_IDX)
    )

    if resultado[
        [
            "true_idx",
            "predicted_idx",
        ]
    ].isna().any().any():
        raise ValueError(
            "Foram produzidos valores ausentes "
            "durante a aplicação do baseline."
        )

    resultado["true_idx"] = (
        resultado["true_idx"]
        .astype(int)
    )

    resultado["predicted_idx"] = (
        resultado["predicted_idx"]
        .astype(int)
    )

    resultado["correct"] = (
        resultado["true_idx"]
        == resultado["predicted_idx"]
    )

    return resultado


def comparar_com_modelos(
    baseline_summary: dict,
    comparison_path: Path,
) -> pd.DataFrame:

    if not comparison_path.exists():
        raise FileNotFoundError(
            comparison_path
        )

    modelos = pd.read_csv(
        comparison_path
    )

    modelos = modelos[
        (
            modelos["level"]
            == "painting"
        )
        & (
            modelos["aggregation"]
            == "mean_probability"
        )
    ][
        [
            "model",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ]
    ].copy()

    baseline = pd.DataFrame(
        [
            {
                "model":
                    "Baseline genre-only",
                "accuracy":
                    baseline_summary[
                        "accuracy"
                    ],
                "macro_precision":
                    baseline_summary[
                        "macro_precision"
                    ],
                "macro_recall":
                    baseline_summary[
                        "macro_recall"
                    ],
                "macro_f1":
                    baseline_summary[
                        "macro_f1"
                    ],
            }
        ]
    )

    return pd.concat(
        [
            baseline,
            modelos,
        ],
        ignore_index=True,
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default=(
            "data/metadata/train.csv"
        ),
    )

    parser.add_argument(
        "--test",
        default=(
            "data/metadata/test.csv"
        ),
    )

    parser.add_argument(
        "--comparison",
        default=(
            "outputs/evaluation/"
            "comparison_summary.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/shortcut_learning"
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = carregar_metadata(
        Path(args.train)
    )

    test_df = carregar_metadata(
        Path(args.test)
    )

    mapa = aprender_mapa_genero(
        train_df
    )

    predicoes = aplicar_baseline(
        test_df,
        mapa,
    )

    resumo, por_classe = (
        calcular_metricas(
            predicoes["true_idx"],
            predicoes[
                "predicted_idx"
            ],
        )
    )

    resumo.update(
        {
            "baseline":
                "genre_majority_train",
            "train_samples":
                int(len(train_df)),
            "test_samples":
                int(len(test_df)),
            "n_genres_train":
                int(
                    train_df[
                        "Genero"
                    ].nunique()
                ),
            "n_genres_test":
                int(
                    test_df[
                        "Genero"
                    ].nunique()
                ),
        }
    )

    mapa.to_csv(
        output_dir
        / "genre_baseline_mapping.csv",
        index=False,
    )

    predicoes.to_csv(
        output_dir
        / "genre_baseline_predictions.csv",
        index=False,
    )

    por_classe.to_csv(
        output_dir
        / "genre_baseline_metrics_by_class.csv",
        index=False,
    )

    with (
        output_dir
        / "genre_baseline_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            resumo,
            arquivo,
            indent=2,
            ensure_ascii=False,
        )

    salvar_matriz_confusao(
        predicoes["true_idx"],
        predicoes["predicted_idx"],
        output_dir
        / "genre_baseline_confusion.png",
        (
            "Baseline usando apenas genre "
            "— conjunto de teste"
        ),
    )

    comparacao = comparar_com_modelos(
        resumo,
        Path(args.comparison),
    )

    comparacao.to_csv(
        output_dir
        / "genre_baseline_vs_models.csv",
        index=False,
    )

    print()
    print(
        "=== Etapa 10C — baseline genre-only ==="
    )

    print()
    print(
        "Regra aprendida exclusivamente "
        "no conjunto de treino:"
    )

    tabela_mapa = mapa.copy()

    tabela_mapa[
        "class_share_train"
    ] *= 100

    print(
        tabela_mapa.round(2)
        .to_string(index=False)
    )

    print()
    print(
        "=== Desempenho no teste ==="
    )

    print(
        f"Accuracy: "
        f"{resumo['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro-Precision: "
        f"{resumo['macro_precision'] * 100:.2f}%"
    )

    print(
        f"Macro-Recall: "
        f"{resumo['macro_recall'] * 100:.2f}%"
    )

    print(
        f"Macro-F1: "
        f"{resumo['macro_f1'] * 100:.2f}%"
    )

    print()
    print(
        "Referência de acaso para "
        "4 classes balanceadas: 25,00%"
    )

    print()
    print(
        "=== Comparação com os modelos ==="
    )

    tabela = comparacao.copy()

    for coluna in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]:
        tabela[coluna] *= 100

    print(
        tabela.round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()