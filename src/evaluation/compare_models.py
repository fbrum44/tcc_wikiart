from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVALUATION_DIR = Path("outputs/evaluation")

MODELOS = {
    "resnet50": "ResNet-50",
    "vit_b16": "ViT-B/16",
}

CENARIOS = [
    (
        "patch",
        "direct",
        "test_patch_summary.json",
        "test_patch_metrics_by_class.csv",
    ),
    (
        "painting",
        "majority",
        "test_painting_majority_summary.json",
        "test_painting_majority_metrics_by_class.csv",
    ),
    (
        "painting",
        "mean_probability",
        "test_painting_mean_probability_summary.json",
        "test_painting_mean_probability_metrics_by_class.csv",
    ),
]


def ler_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        encoding="utf-8"
    ) as arquivo:
        return json.load(arquivo)


def consolidar_resumos() -> pd.DataFrame:
    registros = []

    for model_key, model_name in MODELOS.items():
        model_dir = (
            EVALUATION_DIR
            / model_key
        )

        for (
            nivel,
            agregacao,
            summary_file,
            _,
        ) in CENARIOS:

            resumo = ler_json(
                model_dir
                / summary_file
            )

            registros.append(
                {
                    "model": model_name,
                    "level": nivel,
                    "aggregation": agregacao,
                    "accuracy":
                        resumo["accuracy"],
                    "macro_precision":
                        resumo["macro_precision"],
                    "macro_recall":
                        resumo["macro_recall"],
                    "macro_f1":
                        resumo["macro_f1"],
                    "loss":
                        resumo.get("loss"),
                    "n_samples":
                        resumo["n_samples"],
                }
            )

    return pd.DataFrame(
        registros
    )


def consolidar_metricas_por_classe() -> pd.DataFrame:
    frames = []

    for model_key, model_name in MODELOS.items():

        model_dir = (
            EVALUATION_DIR
            / model_key
        )

        for (
            nivel,
            agregacao,
            _,
            metrics_file,
        ) in CENARIOS:

            path = (
                model_dir
                / metrics_file
            )

            if not path.exists():
                raise FileNotFoundError(
                    path
                )

            df = pd.read_csv(
                path
            )

            df.insert(
                0,
                "aggregation",
                agregacao,
            )

            df.insert(
                0,
                "level",
                nivel,
            )

            df.insert(
                0,
                "model",
                model_name,
            )

            frames.append(
                df
            )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def gerar_grafico_global(
    resumo: pd.DataFrame,
    output_dir: Path,
) -> None:

    labels = [
        "ResNet\nPatch",
        "ResNet\nMaioria",
        "ResNet\nMédia",
        "ViT\nPatch",
        "ViT\nMaioria",
        "ViT\nMédia",
    ]

    accuracy = (
        resumo["accuracy"]
        .to_numpy()
        * 100
    )

    macro_f1 = (
        resumo["macro_f1"]
        .to_numpy()
        * 100
    )

    x = np.arange(
        len(labels)
    )

    largura = 0.36

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    barras_acc = ax.bar(
        x - largura / 2,
        accuracy,
        largura,
        label="Accuracy",
    )

    barras_f1 = ax.bar(
        x + largura / 2,
        macro_f1,
        largura,
        label="Macro-F1",
    )

    ax.set_ylabel(
        "Desempenho (%)"
    )

    ax.set_title(
        "ResNet-50 × ViT-B/16 — conjunto de teste"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for barras in (
        barras_acc,
        barras_f1,
    ):
        ax.bar_label(
            barras,
            fmt="%.1f",
            padding=3,
        )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "comparison_accuracy_macro_f1.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def gerar_grafico_f1_classes(
    por_classe: pd.DataFrame,
    output_dir: Path,
) -> None:

    df = por_classe[
        (
            por_classe["level"]
            == "painting"
        )
        & (
            por_classe["aggregation"]
            == "mean_probability"
        )
    ].copy()

    tabela = df.pivot(
        index="class",
        columns="model",
        values="f1",
    )

    tabela = (
        tabela
        * 100
    )

    ax = tabela.plot.bar(
        figsize=(10, 6)
    )

    ax.set_title(
        "F1 por classe — avaliação por pintura"
    )

    ax.set_xlabel(
        "Movimento artístico"
    )

    ax.set_ylabel(
        "F1-score (%)"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend(
        title="Modelo"
    )

    plt.xticks(
        rotation=0
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "comparison_f1_by_class.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def gerar_curva_loss(
    output_dir: Path,
) -> None:

    resnet = pd.read_csv(
        "outputs/resnet50/"
        "resnet_history.csv"
    )

    vit = pd.read_csv(
        "outputs/vit/"
        "vit_history.csv"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for fase, parte in resnet.groupby(
        "phase",
        sort=False,
    ):
        ax.plot(
            parte["overall_epoch"],
            parte["val_loss"],
            marker="o",
            label=(
                "ResNet-50"
                if fase == "head"
                else None
            ),
        )

    for fase, parte in vit.groupby(
        "phase",
        sort=False,
    ):
        ax.plot(
            parte["epoch_timeline"],
            parte["val_loss"],
            marker="s",
            label=(
                "ViT-B/16"
                if fase == "head"
                else None
            ),
        )

    ax.axvline(
        3.5,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        "Validation loss — ResNet-50 × ViT-B/16"
    )

    ax.set_xlabel(
        "Época (1–3: cabeça | 4–6: fine-tuning)"
    )

    ax.set_ylabel(
        "Validation loss"
    )

    ax.set_xticks(
        range(1, 7)
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "comparison_validation_loss.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumo = (
        consolidar_resumos()
    )

    por_classe = (
        consolidar_metricas_por_classe()
    )

    resumo.to_csv(
        EVALUATION_DIR
        / "comparison_summary.csv",
        index=False,
    )

    por_classe.to_csv(
        EVALUATION_DIR
        / "comparison_by_class.csv",
        index=False,
    )

    gerar_grafico_global(
        resumo,
        EVALUATION_DIR,
    )

    gerar_grafico_f1_classes(
        por_classe,
        EVALUATION_DIR,
    )

    gerar_curva_loss(
        EVALUATION_DIR,
    )

    print()
    print(
        "=== Comparação final — Etapa 9 ==="
    )

    tabela = resumo.copy()

    for coluna in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]:
        tabela[coluna] = (
            tabela[coluna]
            * 100
        )

    print(
        tabela[
            [
                "model",
                "level",
                "aggregation",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "n_samples",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Arquivos gerados:"
    )

    for nome in [
        "comparison_summary.csv",
        "comparison_by_class.csv",
        "comparison_accuracy_macro_f1.png",
        "comparison_f1_by_class.png",
        "comparison_validation_loss.png",
    ]:
        print(
            "-",
            EVALUATION_DIR / nome,
        )


if __name__ == "__main__":
    main()