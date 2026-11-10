from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


from src.data.labels import CLASSES_FINAIS

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

REQUIRED_METADATA_COLUMNS = {
    "Painting_ID",
    "Macro_Estilo",
    "genre_id",
    "Genero",
}

REQUIRED_PREDICTION_COLUMNS = {
    "Painting_ID",
    "true_class",
    "pred_majority_class",
    "pred_mean_class",
}


def carregar_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    faltantes = REQUIRED_METADATA_COLUMNS - set(df.columns)

    if faltantes:
        raise ValueError(
            f"{path} não contém as colunas necessárias: "
            f"{sorted(faltantes)}"
        )

    if df["Painting_ID"].duplicated().any():
        raise ValueError(
            f"{path} contém Painting_ID duplicado."
        )

    return df


def carregar_predicoes(
    path: Path,
    model_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    faltantes = (
        REQUIRED_PREDICTION_COLUMNS
        - set(df.columns)
    )

    if faltantes:
        raise ValueError(
            f"{path} não contém as colunas necessárias: "
            f"{sorted(faltantes)}"
        )

    if df["Painting_ID"].duplicated().any():
        raise ValueError(
            f"{model_name}: Painting_ID duplicado."
        )

    return df[
        [
            "Painting_ID",
            "true_class",
            "pred_majority_class",
            "pred_mean_class",
        ]
    ].rename(
        columns={
            "true_class":
                f"true_class_{model_name}",
            "pred_majority_class":
                f"pred_majority_{model_name}",
            "pred_mean_class":
                f"pred_mean_{model_name}",
        }
    )


def validar_classes(
    df: pd.DataFrame,
    coluna: str,
) -> None:
    classes = set(
        df[coluna].unique()
    )

    esperadas = set(
        CLASSES_FINAIS
    )

    if classes != esperadas:
        raise ValueError(
            f"Classes inesperadas em {coluna}. "
            f"Encontradas: {sorted(classes)}"
        )


def criar_distribuicao_genero(
    df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    tabela = (
        df.groupby(
            [
                "Macro_Estilo",
                "Genero",
            ],
            observed=True,
        )
        .size()
        .reset_index(
            name="n"
        )
    )

    tabela["percent_class"] = (
        tabela["n"]
        / tabela.groupby(
            "Macro_Estilo"
        )["n"].transform("sum")
        * 100
    )

    tabela["split"] = split_name

    tabela = tabela[
        [
            "split",
            "Macro_Estilo",
            "Genero",
            "n",
            "percent_class",
        ]
    ]

    return tabela.sort_values(
        [
            "Macro_Estilo",
            "n",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


def preparar_base_teste(
    test_df: pd.DataFrame,
    resnet_df: pd.DataFrame,
    vit_df: pd.DataFrame,
) -> pd.DataFrame:

    base = test_df[
        [
            "Painting_ID",
            "Macro_Estilo",
            "genre_id",
            "Genero",
        ]
    ].copy()

    base = base.merge(
        resnet_df,
        on="Painting_ID",
        how="left",
        validate="one_to_one",
    )

    base = base.merge(
        vit_df,
        on="Painting_ID",
        how="left",
        validate="one_to_one",
    )

    colunas_predicao = [
        "true_class_resnet50",
        "pred_majority_resnet50",
        "pred_mean_resnet50",
        "true_class_vit_b16",
        "pred_majority_vit_b16",
        "pred_mean_vit_b16",
    ]

    if base[colunas_predicao].isna().any().any():
        raise ValueError(
            "Existem pinturas de teste sem predição."
        )

    if len(base) != 1500:
        raise ValueError(
            "A base integrada deve conter "
            f"1.500 pinturas; encontrou {len(base)}."
        )

    if base["Painting_ID"].nunique() != 1500:
        raise ValueError(
            "Painting_ID não é único na base integrada."
        )

    if not (
        base["Macro_Estilo"]
        == base["true_class_resnet50"]
    ).all():
        raise ValueError(
            "Rótulos reais da ResNet não coincidem "
            "com test.csv."
        )

    if not (
        base["Macro_Estilo"]
        == base["true_class_vit_b16"]
    ).all():
        raise ValueError(
            "Rótulos reais do ViT não coincidem "
            "com test.csv."
        )

    validar_classes(
        base,
        "Macro_Estilo",
    )

    return base

def calcular_desempenho_por_genero(
    base: pd.DataFrame,
    *,
    model_name: str,
    pred_column: str,
) -> pd.DataFrame:

    registros = []

    for genero, grupo in base.groupby(
        "Genero",
        sort=True,
    ):
        y_true = grupo[
            "Macro_Estilo"
        ]

        y_pred = grupo[
            pred_column
        ]

        classes_presentes = sorted(
            y_true.unique()
        )

        registros.append(
            {
                "model": model_name,
                "Genero": genero,
                "n": len(grupo),
                "n_classes_present":
                    len(classes_presentes),
                "accuracy":
                    accuracy_score(
                        y_true,
                        y_pred,
                    ),
                "macro_f1_present_classes":
                    f1_score(
                        y_true,
                        y_pred,
                        labels=classes_presentes,
                        average="macro",
                        zero_division=0,
                    ),
            }
        )

    return pd.DataFrame(
        registros
    ).sort_values(
        [
            "n",
            "Genero",
        ],
        ascending=[
            False,
            True,
        ],
    )


def calcular_desempenho_classe_genero(
    base: pd.DataFrame,
    *,
    model_name: str,
    pred_column: str,
) -> pd.DataFrame:

    registros = []

    for (
        classe,
        genero,
    ), grupo in base.groupby(
        [
            "Macro_Estilo",
            "Genero",
        ],
        sort=True,
    ):

        corretos = (
            grupo[pred_column]
            == grupo["Macro_Estilo"]
        )

        registros.append(
            {
                "model": model_name,
                "Macro_Estilo": classe,
                "Genero": genero,
                "n": len(grupo),
                "correct": int(
                    corretos.sum()
                ),
                "recall_within_genre":
                    float(
                        corretos.mean()
                    ),
            }
        )

    return pd.DataFrame(
        registros
    ).sort_values(
        [
            "Macro_Estilo",
            "n",
        ],
        ascending=[
            True,
            False,
        ],
    )

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default="data/metadata/train.csv",
    )

    parser.add_argument(
        "--test",
        default="data/metadata/test.csv",
    )

    parser.add_argument(
        "--resnet-predictions",
        default=(
            "outputs/evaluation/resnet50/"
            "test_painting_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--vit-predictions",
        default=(
            "outputs/evaluation/vit_b16/"
            "test_painting_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/shortcut_learning",
    )

    args = parser.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = carregar_metadata(
        train_path
    )

    test_df = carregar_metadata(
        test_path
    )

    validar_classes(
        train_df,
        "Macro_Estilo",
    )

    validar_classes(
        test_df,
        "Macro_Estilo",
    )

    resnet_df = carregar_predicoes(
        Path(
            args.resnet_predictions
        ),
        "resnet50",
    )

    vit_df = carregar_predicoes(
        Path(
            args.vit_predictions
        ),
        "vit_b16",
    )

    base_teste = preparar_base_teste(
        test_df,
        resnet_df,
        vit_df,
    )

    desempenho_genero_resnet = (
    calcular_desempenho_por_genero(
        base_teste,
        model_name="ResNet-50",
        pred_column="pred_mean_resnet50",
    )
)

    desempenho_genero_vit = (
        calcular_desempenho_por_genero(
            base_teste,
            model_name="ViT-B/16",
            pred_column="pred_mean_vit_b16",
        )
    )

    desempenho_genero = pd.concat(
        [
            desempenho_genero_resnet,
            desempenho_genero_vit,
        ],
        ignore_index=True,
    )


    classe_genero_resnet = (
        calcular_desempenho_classe_genero(
            base_teste,
            model_name="ResNet-50",
            pred_column="pred_mean_resnet50",
        )
    )

    classe_genero_vit = (
        calcular_desempenho_classe_genero(
            base_teste,
            model_name="ViT-B/16",
            pred_column="pred_mean_vit_b16",
        )
    )

    desempenho_classe_genero = (
            pd.concat(
                [
                    classe_genero_resnet,
                    classe_genero_vit,
                ],
                ignore_index=True,
            )
    )

    distribuicao_train = (
            criar_distribuicao_genero(
                train_df,
                "train",
            )
        )

    distribuicao_test = (
        criar_distribuicao_genero(
            test_df,
            "test",
        )
    )

    base_teste.to_csv(
        output_dir
        / "test_genre_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    distribuicao_train.to_csv(
        output_dir
        / "genre_distribution_train.csv",
        index=False,
    )

    distribuicao_test.to_csv(
        output_dir
        / "genre_distribution_test.csv",
        index=False,
    )

    print()
    print(
        "=== Etapa 10A — preparação concluída ==="
    )

    print(
        "Treino:",
        len(train_df),
        "pinturas",
    )

    print(
        "Teste:",
        len(test_df),
        "pinturas",
    )

    print(
        "Base integrada:",
        len(base_teste),
        "pinturas",
    )

    print(
        "Gêneros no treino:",
        train_df["Genero"].nunique(),
    )

    print(
        "Gêneros no teste:",
        test_df["Genero"].nunique(),
    )

    print()
    print(
        "Unknown Genre no treino:",
        (
            train_df["Genero"]
            == "Unknown Genre"
        ).sum(),
    )

    print(
        "Unknown Genre no teste:",
        (
            test_df["Genero"]
            == "Unknown Genre"
        ).sum(),
    )

    print()
    print(
        "Distribuição de gênero "
        "por macroclasse no teste:"
    )

    tabela = pd.crosstab(
        test_df["Macro_Estilo"],
        test_df["Genero"],
        normalize="index",
    ) * 100

    tabela = tabela.reindex(
        CLASSES_FINAIS
    )

    print(
        tabela.round(2).to_string()
    )

    print()
    print(
        "Arquivos gerados em:",
        output_dir,
    )

    desempenho_genero.to_csv(
        output_dir
        / "performance_by_genre.csv",
        index=False,
    )

    desempenho_classe_genero.to_csv(
        output_dir
        / "performance_by_class_genre.csv",
        index=False,
    )

    print()
    print(
        "=== Desempenho por gênero ==="
    )

    for modelo in [
        "ResNet-50",
        "ViT-B/16",
    ]:
        print()
        print(modelo)

        tabela = desempenho_genero[
            desempenho_genero["model"]
            == modelo
        ].copy()

        tabela["accuracy"] *= 100
        tabela[
            "macro_f1_present_classes"
        ] *= 100

        print(
            tabela.round(2).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
