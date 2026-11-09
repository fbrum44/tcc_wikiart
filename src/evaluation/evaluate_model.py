from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from src.data.labels import CLASSES_FINAIS
from src.data.normalization import (
    HALF_MEAN,
    HALF_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.data.patch_dataset import (
    PatchDataset,
    WikiArtImageSource,
    load_patch_manifest,
)
from src.models.resnet import criar_resnet50
from src.models.vit import criar_vit


PROB_COLUMNS = [
    f"prob_{classe}"
    for classe in CLASSES_FINAIS
]


def carregar_modelo(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict, tuple, tuple]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_classes = checkpoint.get("classes")

    if list(checkpoint_classes) != list(CLASSES_FINAIS):
        raise ValueError(
            "As classes do checkpoint não coincidem "
            "com CLASSES_FINAIS."
        )

    if model_name == "resnet50":
        modelo = criar_resnet50(
            usar_pesos_pre_treinados=False,
            congelar_backbone_inicialmente=False,
        )
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

    elif model_name == "vit_b16":
        modelo = criar_vit(
            usar_pesos_pre_treinados=False,
            congelar_backbone_inicialmente=False,
        )
        mean = HALF_MEAN
        std = HALF_STD

    else:
        raise ValueError(
            f"Modelo desconhecido: {model_name}"
        )

    modelo.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    modelo = modelo.to(device)
    modelo.eval()

    return modelo, checkpoint, mean, std


def criar_loader(
    split: str,
    *,
    mean,
    std,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, PatchDataset]:
    manifest_path = Path(
        f"data/metadata/patches_{split}.csv.gz"
    )

    manifest = load_patch_manifest(
        manifest_path
    )

    dataset = PatchDataset(
        manifest,
        WikiArtImageSource(),
        is_train=False,
        augment=None,
        mean=mean,
        std=std,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(
            num_workers > 0
        ),
        pin_memory=torch.cuda.is_available(),
    )

    return loader, dataset


def executar_inferencia(
    modelo: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[pd.DataFrame, float]:
    criterio = nn.CrossEntropyLoss(
        reduction="sum"
    )

    registros = []

    soma_loss = 0.0
    total = 0

    total_batches = len(loader)

    with torch.inference_mode():

        for batch_idx, batch in enumerate(
            loader,
            start=1,
        ):
            if (
                max_batches is not None
                and batch_idx > max_batches
            ):
                break

            imagens = batch["image"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            logits = modelo(imagens)

            loss = criterio(
                logits,
                labels,
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            predicoes = probs.argmax(
                dim=1
            )

            soma_loss += loss.item()
            total += labels.size(0)

            probs_np = (
                probs
                .detach()
                .cpu()
                .numpy()
            )

            labels_np = (
                labels
                .detach()
                .cpu()
                .numpy()
            )

            pred_np = (
                predicoes
                .detach()
                .cpu()
                .numpy()
            )

            for i in range(
                len(labels_np)
            ):
                registro = {
                    "Patch_ID":
                        batch["patch_id"][i],
                    "Painting_ID":
                        batch["painting_id"][i],
                    "x0":
                        int(batch["x0"][i]),
                    "y0":
                        int(batch["y0"][i]),
                    "true_idx":
                        int(labels_np[i]),
                    "pred_idx":
                        int(pred_np[i]),
                    "true_class":
                        CLASSES_FINAIS[
                            int(labels_np[i])
                        ],
                    "pred_class":
                        CLASSES_FINAIS[
                            int(pred_np[i])
                        ],
                }

                for class_idx, coluna in enumerate(
                    PROB_COLUMNS
                ):
                    registro[coluna] = float(
                        probs_np[
                            i,
                            class_idx,
                        ]
                    )

                registros.append(
                    registro
                )

            if (
                batch_idx == 1
                or batch_idx % 100 == 0
                or batch_idx == total_batches
            ):
                print(
                    f"Inferência: "
                    f"{batch_idx}/"
                    f"{total_batches} batches"
                )

    if total == 0:
        raise RuntimeError(
            "Nenhum patch foi processado."
        )

    df = pd.DataFrame(
        registros
    )

    loss_media = (
        soma_loss / total
    )

    return df, loss_media


def calcular_metricas(
    y_true,
    y_pred,
) -> tuple[dict, pd.DataFrame]:
    labels = list(
        range(len(CLASSES_FINAIS))
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    resumo = {
        "accuracy": float(
            accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "n_samples": int(
            len(y_true)
        ),
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


def salvar_matriz_confusao(
    y_true,
    y_pred,
    output_path: Path,
    title: str,
) -> None:
    matriz = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(len(CLASSES_FINAIS))
        ),
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
        rotation=45,
        ha="right",
    )

    limite = (
        matriz.max() / 2.0
        if matriz.size
        else 0
    )

    for i in range(
        matriz.shape[0]
    ):
        for j in range(
            matriz.shape[1]
        ):
            ax.text(
                j,
                i,
                str(
                    matriz[i, j]
                ),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matriz[i, j] > limite
                    else "black"
                ),
            )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet50",
            "vit_b16",
        ],
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "test",
        ],
        default="validation",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if (
        args.split == "test"
        and args.max_batches is not None
    ):
        raise ValueError(
            "Não é permitido executar avaliação "
            "parcial no conjunto de teste."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Dispositivo:",
        device,
    )

    modelo, checkpoint, mean, std = (
        carregar_modelo(
            args.model,
            Path(args.checkpoint),
            device,
        )
    )

    print(
        "Modelo:",
        args.model,
    )

    print(
        "Checkpoint validation loss:",
        checkpoint.get(
            "val_loss"
        ),
    )

    loader, dataset = criar_loader(
        args.split,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(
        "Split:",
        args.split,
    )

    print(
        "Patches disponíveis:",
        len(dataset),
    )

    predicoes, loss = executar_inferencia(
        modelo,
        loader,
        device,
        max_batches=args.max_batches,
    )

    resumo, por_classe = (
        calcular_metricas(
            predicoes["true_idx"],
            predicoes["pred_idx"],
        )
    )

    resumo["loss"] = float(
        loss
    )

    resumo["model"] = (
        args.model
    )

    resumo["split"] = (
        args.split
    )

    if args.max_batches is None:
        sufixo = args.split
    else:
        sufixo = (
            f"{args.split}_smoke"
        )

    output_dir = (
        Path("outputs")
        / "evaluation"
        / args.model
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_dir
        / f"{sufixo}_patch_predictions.csv.gz"
    )

    summary_path = (
        output_dir
        / f"{sufixo}_patch_summary.json"
    )

    class_metrics_path = (
        output_dir
        / f"{sufixo}_patch_metrics_by_class.csv"
    )

    confusion_path = (
        output_dir
        / f"{sufixo}_patch_confusion.png"
    )

    predicoes.to_csv(
        predictions_path,
        index=False,
        compression="gzip",
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            resumo,
            arquivo,
            indent=2,
            ensure_ascii=False,
        )

    por_classe.to_csv(
        class_metrics_path,
        index=False,
    )

    salvar_matriz_confusao(
        predicoes["true_idx"],
        predicoes["pred_idx"],
        confusion_path,
        (
            f"{args.model} — "
            f"{args.split} — patches"
        ),
    )

    print()
    print(
        "=== Avaliação concluída ==="
    )

    print(
        json.dumps(
            resumo,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "Predições:",
        predictions_path,
    )

    print(
        "Métricas por classe:",
        class_metrics_path,
    )

    print(
        "Matriz de confusão:",
        confusion_path,
    )


if __name__ == "__main__":
    main()