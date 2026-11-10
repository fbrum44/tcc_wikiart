from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.data.labels import CLASSES_FINAIS
from src.evaluation.shortcut_analysis import (
    carregar_metadata,
    carregar_predicoes,
    preparar_base_teste,
)


def calcular_metricas(
    df: pd.DataFrame,
    pred_column: str,
) -> tuple[dict, pd.DataFrame]:

    y_true = df["Macro_Estilo"]
    y_pred = df[pred_column]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASSES_FINAIS,
        average="macro",
        zero_division=0,
    )

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES_FINAIS,
        zero_division=0,
    )

    resumo = {
        "n": len(df),
        "accuracy": accuracy,
        "balanced_accuracy":
            balanced_accuracy,
        "macro_f1": macro_f1,
    }

    por_classe = pd.DataFrame(
        {
            "class": CLASSES_FINAIS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    return resumo, por_classe


def salvar_matriz_confusao_normalizada(
    df: pd.DataFrame,
    pred_column: str,
    output_path: Path,
    title: str,
) -> None:

    matriz = confusion_matrix(
        df["Macro_Estilo"],
        df[pred_column],
        labels=CLASSES_FINAIS,
        normalize="true",
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    imagem = ax.imshow(
        matriz,
        interpolation="nearest",
    )

    fig.colorbar(
        imagem,
        ax=ax,
    )

    ax.set(
        xticks=np.arange(
            len(CLASSES_FINAIS)
        ),
        yticks=np.arange(
            len(CLASSES_FINAIS)
        ),
        xticklabels=CLASSES_FINAIS,
        yticklabels=CLASSES_FINAIS,
        xlabel="Classe predita",
        ylabel="Classe real",
        title=title,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
    )

    limite = (
        matriz.max() / 2
        if matriz.size
        else 0.5
    )

    for i in range(
        matriz.shape[0]
    ):
        for j in range(
            matriz.shape[1]
        ):
            valor = matriz[i, j]

            ax.text(
                j,
                i,
                f"{valor * 100:.1f}%",
                ha="center",
                va="center",
                color=(
                    "white"
                    if valor > limite
                    else "black"
                ),
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        default=(
            "data/metadata/test.csv"
        ),
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
        "--genre",
        default="portrait",
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

    test_df = carregar_metadata(
        Path(args.test)
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

    base = preparar_base_teste(
        test_df,
        resnet_df,
        vit_df,
    )

    controlado = base[
        base["Genero"]
        == args.genre
    ].copy()

    if controlado.empty:
        raise ValueError(
            f"Nenhuma pintura do gênero "
            f"{args.genre!r} foi encontrada."
        )

    classes_presentes = set(
        controlado[
            "Macro_Estilo"
        ].unique()
    )

    if classes_presentes != set(
        CLASSES_FINAIS
    ):
        raise ValueError(
            "O subconjunto controlado não "
            "contém todas as quatro classes. "
            f"Encontradas: "
            f"{sorted(classes_presentes)}"
        )

    contagem_classes = (
        controlado[
            "Macro_Estilo"
        ]
        .value_counts()
        .reindex(
            CLASSES_FINAIS
        )
        .rename("n")
        .reset_index()
        .rename(
            columns={
                "Macro_Estilo":
                    "class"
            }
        )
    )

    modelos = [
        (
            "ResNet-50",
            "pred_mean_resnet50",
        ),
        (
            "ViT-B/16",
            "pred_mean_vit_b16",
        ),
    ]

    resumos = []
    resultados_classe = []

    for (
        model_name,
        pred_column,
    ) in modelos:

        resumo_full, _ = (
            calcular_metricas(
                base,
                pred_column,
            )
        )

        (
            resumo_controlado,
            por_classe,
        ) = calcular_metricas(
            controlado,
            pred_column,
        )

        resumos.append(
            {
                "model": model_name,
                "scenario":
                    "full_test",
                **resumo_full,
            }
        )

        resumos.append(
            {
                "model": model_name,
                "scenario":
                    args.genre,
                **resumo_controlado,
            }
        )

        por_classe.insert(
            0,
            "model",
            model_name,
        )

        por_classe.insert(
            1,
            "genre",
            args.genre,
        )

        resultados_classe.append(
            por_classe
        )

        nome_modelo = (
            model_name
            .lower()
            .replace("-", "")
            .replace("/", "")
        )

        salvar_matriz_confusao_normalizada(
            controlado,
            pred_column,
            output_dir
            / (
                f"{args.genre}_"
                f"{nome_modelo}_"
                "confusion_normalized.png"
            ),
            (
                f"{model_name} — "
                f"{args.genre} "
                "(matriz normalizada)"
            ),
        )

    resumo_df = pd.DataFrame(
        resumos
    )

    por_classe_df = pd.concat(
        resultados_classe,
        ignore_index=True,
    )

    comparacao = []

    for model_name, _ in modelos:

        modelo = resumo_df[
            resumo_df["model"]
            == model_name
        ]

        full = modelo[
            modelo["scenario"]
            == "full_test"
        ].iloc[0]

        portrait = modelo[
            modelo["scenario"]
            == args.genre
        ].iloc[0]

        comparacao.append(
            {
                "model": model_name,
                "full_accuracy":
                    full["accuracy"],
                "portrait_accuracy":
                    portrait["accuracy"],
                "delta_accuracy":
                    (
                        portrait["accuracy"]
                        - full["accuracy"]
                    ),
                "full_balanced_accuracy":
                    full[
                        "balanced_accuracy"
                    ],
                "portrait_balanced_accuracy":
                    portrait[
                        "balanced_accuracy"
                    ],
                "delta_balanced_accuracy":
                    (
                        portrait[
                            "balanced_accuracy"
                        ]
                        - full[
                            "balanced_accuracy"
                        ]
                    ),
                "full_macro_f1":
                    full["macro_f1"],
                "portrait_macro_f1":
                    portrait["macro_f1"],
                "delta_macro_f1":
                    (
                        portrait["macro_f1"]
                        - full["macro_f1"]
                    ),
            }
        )

    comparacao_df = pd.DataFrame(
        comparacao
    )

    controlado.to_csv(
        output_dir
        / "portrait_control_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    contagem_classes.to_csv(
        output_dir
        / "portrait_control_class_counts.csv",
        index=False,
    )

    resumo_df.to_csv(
        output_dir
        / "portrait_control_metrics.csv",
        index=False,
    )

    por_classe_df.to_csv(
        output_dir
        / "portrait_control_metrics_by_class.csv",
        index=False,
    )

    comparacao_df.to_csv(
        output_dir
        / "portrait_control_comparison.csv",
        index=False,
    )

    print()
    print(
        "=== Etapa 10D — "
        "subconjunto controlado portrait ==="
    )

    print()
    print(
        "Total de pinturas:",
        len(controlado),
    )

    print()
    print(
        "Distribuição por classe:"
    )

    print(
        contagem_classes
        .to_string(index=False)
    )

    print()
    print(
        "=== Comparação teste completo "
        "vs portrait ==="
    )

    tabela = comparacao_df.copy()

    metricas = [
        "full_accuracy",
        "portrait_accuracy",
        "delta_accuracy",
        "full_balanced_accuracy",
        "portrait_balanced_accuracy",
        "delta_balanced_accuracy",
        "full_macro_f1",
        "portrait_macro_f1",
        "delta_macro_f1",
    ]

    for coluna in metricas:
        tabela[coluna] *= 100

    print(
        tabela.round(2)
        .to_string(index=False)
    )

    print()
    print(
        "=== Métricas por classe "
        "no subconjunto portrait ==="
    )

    tabela_classe = (
        por_classe_df.copy()
    )

    for coluna in [
        "precision",
        "recall",
        "f1",
    ]:
        tabela_classe[coluna] *= 100

    print(
        tabela_classe
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()