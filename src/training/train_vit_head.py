"""Etapa 8 - Treinamento completo da cabeça classificadora do ViT-B/16.

Espelha src/training/train_resnet_head.py: mesma configuração, mesmo amostrador de
validação (patches fixos por pintura, mesma seed), mesmo otimizador (AdamW) e mesmo
critério de checkpoint (menor validation loss). Reaproveita dele as funções genéricas.
"""

from __future__ import annotations

import argparse
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import timm
import torch
from torch import nn

from src.data.labels import CLASSES_FINAIS
from src.data.normalization import HALF_MEAN, HALF_STD
from src.data.patch_dataset import WikiArtImageSource, build_loaders
from src.models.vit import (
    INPUT_SIZE,
    VIT_MODEL_NAME,
    VIT_WEIGHTS,
    criar_vit,
)
from src.training.train_resnet import configurar_seed, obter_dispositivo
from src.training.train_resnet_head import (
    criar_loader_validacao_fixo,
    salvar_configuracao,
    salvar_historico,
    salvar_melhor_checkpoint,
    validar_epoca,
)


@dataclass(frozen=True)
class HeadTrainingConfig:
    seed: int = 42

    epochs: int = 3
    batch_size: int = 32

    train_patches_per_painting: int = 8
    val_patches_per_painting: int = 8

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    num_workers: int = 0

    architecture: str = "vit_base_patch16"
    weights: str = VIT_WEIGHTS
    timm_model: str = VIT_MODEL_NAME

    input_size: int = INPUT_SIZE


def configurar_modo_cabeca_vit(modelo: nn.Module) -> None:
    """Backbone em eval (equivale ao BatchNorm congelado da ResNet); só a cabeça em treino."""

    modelo.eval()
    modelo.head.train()


def treinar_epoca(modelo, loader, criterio, otimizador, dispositivo) -> tuple[float, float]:

    configurar_modo_cabeca_vit(modelo)

    soma_loss = 0.0
    corretos = 0
    total = 0
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):

        imagens = batch["image"].to(dispositivo, non_blocking=True)
        labels = batch["label"].to(dispositivo, non_blocking=True)

        otimizador.zero_grad(set_to_none=True)

        logits = modelo(imagens)
        loss = criterio(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss inválida encontrada: {loss.item()}")

        loss.backward()
        otimizador.step()

        tamanho_batch = labels.size(0)
        soma_loss += loss.item() * tamanho_batch
        corretos += (logits.argmax(dim=1) == labels).sum().item()
        total += tamanho_batch

        if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total_batches:
            print(f"  treino: {batch_idx}/{total_batches} batches")

    if total == 0:
        raise RuntimeError("Nenhuma amostra processada no treino.")

    return soma_loss / total, corretos / total


def criar_registro_configuracao(config: HeadTrainingConfig, dispositivo: torch.device) -> dict:

    registro = asdict(config)
    registro.update(
        {
            "classes": CLASSES_FINAIS,
            "normalization_mean": list(HALF_MEAN),
            "normalization_std": list(HALF_STD),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "timm_version": timm.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(dispositivo),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "validation_strategy": "fixed_patches_per_painting",
            "checkpoint_metric": "validation_loss",
            "phase": "classifier_head_only",
        }
    )
    return registro


def executar_treinamento(config: HeadTrainingConfig) -> None:

    print("=== Etapa 8 — Treinamento da cabeça ViT-B/16 ===")

    configurar_seed(config.seed)
    dispositivo = obter_dispositivo()
    print("Dispositivo:", dispositivo)

    loaders = build_loaders(
        manifest_dir="data/metadata",
        image_source=WikiArtImageSource(),
        use_augmentation=True,
        batch_size=config.batch_size,
        patches_per_painting=config.train_patches_per_painting,
        num_workers=config.num_workers,
        mean=HALF_MEAN,
        std=HALF_STD,
        seed=config.seed,
    )

    val_loader = criar_loader_validacao_fixo(
        loaders["validation"],
        patches_per_painting=config.val_patches_per_painting,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed + 1000,
    )

    print("Patches de treino por época:", len(loaders["train"].sampler))
    print("Patches de validação por época:", len(val_loader.sampler))

    modelo = criar_vit(
        usar_pesos_pre_treinados=True,
        congelar_backbone_inicialmente=True,
    ).to(dispositivo)

    treinaveis = [n for n, p in modelo.named_parameters() if p.requires_grad]
    if not all(n.startswith("head.") for n in treinaveis):
        raise AssertionError("Existem parâmetros treináveis fora da cabeça classificadora.")

    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.AdamW(
        modelo.head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history_path = Path("outputs/vit/head_history.csv")
    config_path = Path("outputs/vit/head_config.json")
    checkpoint_path = Path("models/vit_head_best.pt")

    registro_config = criar_registro_configuracao(config, dispositivo)
    salvar_configuracao(registro_config, config_path)

    historico = []
    melhor_val_loss = float("inf")
    melhor_epoca = None

    for epoch in range(1, config.epochs + 1):

        print()
        print(f"===== Época {epoch}/{config.epochs} =====")

        train_loss, train_acc = treinar_epoca(
            modelo, loaders["train"], criterio, otimizador, dispositivo
        )
        val_loss, val_acc = validar_epoca(modelo, val_loader, criterio, dispositivo)

        eh_melhor = val_loss < melhor_val_loss

        historico.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "best": eh_melhor,
            }
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

        if eh_melhor:
            melhor_val_loss = val_loss
            melhor_epoca = epoch

            salvar_melhor_checkpoint(
                checkpoint_path,
                modelo,
                otimizador,
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                config=registro_config,
                historico=historico,
            )
            print("Novo melhor checkpoint salvo.")

        salvar_historico(historico, history_path)

    if not checkpoint_path.exists():
        raise AssertionError("Nenhum checkpoint foi salvo.")

    print()
    print("=== Treinamento da cabeça concluído ===")
    print("Melhor época:", melhor_epoca)
    print("Melhor validation loss:", f"{melhor_val_loss:.4f}")
    print("Histórico:", history_path)
    print("Configuração:", config_path)
    print("Melhor checkpoint:", checkpoint_path)


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-k", type=int, default=8)
    parser.add_argument("--val-k", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    executar_treinamento(
        HeadTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_patches_per_painting=args.train_k,
            val_patches_per_painting=args.val_k,
            num_workers=args.num_workers,
        )
    )


if __name__ == "__main__":
    main()
