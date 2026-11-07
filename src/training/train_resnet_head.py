"""Etapa 7 - Treinamento completo da cabeça classificadora da ResNet-50."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Sampler

from src.data.labels import CLASSES_FINAIS
from src.data.normalization import IMAGENET_MEAN, IMAGENET_STD
from src.data.patch_dataset import (
    WikiArtImageSource,
    build_loaders,
)
from src.models.resnet import (
    RESNET50_WEIGHTS,
    criar_resnet50,
)
from src.training.train_resnet import (
    configurar_modo_cabeca,
    configurar_seed,
    obter_dispositivo,
)


# =========================================================
# Configuração experimental
# =========================================================

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

    architecture: str = "resnet50"
    weights: str = "IMAGENET1K_V2"

    input_size: int = 256


# =========================================================
# Validação balanceada e fixa
# =========================================================

class FixedPaintingSampler(Sampler[int]):
    """
    Seleciona deterministicamente k patches por pintura.

    Diferentemente do sampler de treino:
    - não muda entre épocas;
    - não aplica augmentation;
    - dá o mesmo peso amostral a cada pintura.
    """

    def __init__(
        self,
        painting_ids,
        patches_per_painting: int,
        seed: int,
    ) -> None:

        if patches_per_painting <= 0:
            raise ValueError(
                "patches_per_painting deve ser maior que zero."
            )

        painting_ids = np.asarray(
            painting_ids
        )

        codes, _ = pd.factorize(
            painting_ids
        )

        order = np.argsort(
            codes,
            kind="stable",
        )

        bounds = (
            np.flatnonzero(
                np.diff(codes[order])
            )
            + 1
        )

        groups = np.split(
            order,
            bounds,
        )

        rng = np.random.default_rng(
            seed
        )

        indices = []

        for group in groups:

            quantidade = min(
                patches_per_painting,
                len(group),
            )

            selecionados = rng.choice(
                group,
                size=quantidade,
                replace=False,
            )

            indices.extend(
                selecionados.tolist()
            )

        self.indices = indices

    def __iter__(self):
        return iter(
            self.indices
        )

    def __len__(self) -> int:
        return len(
            self.indices
        )


# =========================================================
# Métricas de uma época
# =========================================================

def treinar_epoca(
    modelo: nn.Module,
    loader,
    criterio: nn.Module,
    otimizador,
    dispositivo: torch.device,
) -> tuple[float, float]:

    # Mantém backbone e BatchNorm congelados.
    configurar_modo_cabeca(
        modelo
    )

    soma_loss = 0.0
    corretos = 0
    total = 0

    total_batches = len(
        loader
    )

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):

        imagens = batch["image"].to(
            dispositivo,
            non_blocking=True,
        )

        labels = batch["label"].to(
            dispositivo,
            non_blocking=True,
        )

        otimizador.zero_grad(
            set_to_none=True
        )

        logits = modelo(
            imagens
        )

        loss = criterio(
            logits,
            labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss inválida encontrada: {loss.item()}"
            )

        loss.backward()

        otimizador.step()

        tamanho_batch = labels.size(0)

        soma_loss += (
            loss.item()
            * tamanho_batch
        )

        predicoes = logits.argmax(
            dim=1
        )

        corretos += (
            predicoes == labels
        ).sum().item()

        total += tamanho_batch

        if (
            batch_idx == 1
            or batch_idx % 25 == 0
            or batch_idx == total_batches
        ):
            print(
                f"  treino: "
                f"{batch_idx}/{total_batches} batches"
            )

    if total == 0:
        raise RuntimeError(
            "Nenhuma amostra processada no treino."
        )

    return (
        soma_loss / total,
        corretos / total,
    )


def validar_epoca(
    modelo: nn.Module,
    loader,
    criterio: nn.Module,
    dispositivo: torch.device,
) -> tuple[float, float]:

    modelo.eval()

    soma_loss = 0.0
    corretos = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            imagens = batch["image"].to(
                dispositivo,
                non_blocking=True,
            )

            labels = batch["label"].to(
                dispositivo,
                non_blocking=True,
            )

            logits = modelo(
                imagens
            )

            loss = criterio(
                logits,
                labels,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Loss de validação inválida: "
                    f"{loss.item()}"
                )

            tamanho_batch = (
                labels.size(0)
            )

            soma_loss += (
                loss.item()
                * tamanho_batch
            )

            predicoes = logits.argmax(
                dim=1
            )

            corretos += (
                predicoes == labels
            ).sum().item()

            total += tamanho_batch

    if total == 0:
        raise RuntimeError(
            "Nenhuma amostra processada "
            "na validação."
        )

    return (
        soma_loss / total,
        corretos / total,
    )


# =========================================================
# Histórico
# =========================================================

def salvar_historico(
    historico: list[dict],
    caminho: Path,
) -> None:

    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    campos = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "best",
    ]

    with caminho.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:

        writer = csv.DictWriter(
            arquivo,
            fieldnames=campos,
        )

        writer.writeheader()

        writer.writerows(
            historico
        )


# =========================================================
# Configuração / ambiente
# =========================================================

def criar_registro_configuracao(
    config: HeadTrainingConfig,
    dispositivo: torch.device,
) -> dict:

    registro = asdict(
        config
    )

    registro.update(
        {
            "classes": CLASSES_FINAIS,
            "normalization_mean": list(
                IMAGENET_MEAN
            ),
            "normalization_std": list(
                IMAGENET_STD
            ),
            "weights_enum": str(
                RESNET50_WEIGHTS
            ),
            "python_version": (
                platform.python_version()
            ),
            "torch_version": (
                torch.__version__
            ),
            "torchvision_version": (
                torchvision.__version__
            ),
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "device": str(
                dispositivo
            ),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "validation_strategy": (
                "fixed_patches_per_painting"
            ),
            "checkpoint_metric": (
                "validation_loss"
            ),
            "phase": (
                "classifier_head_only"
            ),
        }
    )

    return registro


def salvar_configuracao(
    registro: dict,
    caminho: Path,
) -> None:

    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with caminho.open(
        "w",
        encoding="utf-8",
    ) as arquivo:

        json.dump(
            registro,
            arquivo,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# Checkpoint
# =========================================================

def salvar_melhor_checkpoint(
    caminho: Path,
    modelo: nn.Module,
    otimizador,
    *,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
    config: dict,
    historico: list[dict],
) -> None:

    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "phase": "classifier_head_only",
            "epoch": epoch,
            "classes": CLASSES_FINAIS,
            "model_state_dict": (
                modelo.state_dict()
            ),
            "optimizer_state_dict": (
                otimizador.state_dict()
            ),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "config": config,
            "history": historico,
        },
        caminho,
    )


# =========================================================
# Loader de validação fixo
# =========================================================

def criar_loader_validacao_fixo(
    loader_original,
    *,
    patches_per_painting: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:

    dataset = (
        loader_original.dataset
    )

    sampler = FixedPaintingSampler(
        dataset.painting_id,
        patches_per_painting,
        seed,
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=(
            loader_original.worker_init_fn
        ),
        persistent_workers=(
            num_workers > 0
        ),
        generator=generator,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )


# =========================================================
# Experimento
# =========================================================

def executar_treinamento(
    config: HeadTrainingConfig,
) -> None:

    print(
        "=== Etapa 7 — Treinamento da cabeça ResNet-50 ==="
    )

    configurar_seed(
        config.seed
    )

    dispositivo = obter_dispositivo()

    print(
        "Dispositivo:",
        dispositivo,
    )

    # -----------------------------------------------------
    # Dados
    # -----------------------------------------------------

    image_source = (
        WikiArtImageSource()
    )

    loaders = build_loaders(
        manifest_dir="data/metadata",
        image_source=image_source,
        use_augmentation=True,
        batch_size=config.batch_size,
        patches_per_painting=(
            config.train_patches_per_painting
        ),
        num_workers=config.num_workers,
        seed=config.seed,
    )

    val_loader = (
        criar_loader_validacao_fixo(
            loaders["validation"],
            patches_per_painting=(
                config.val_patches_per_painting
            ),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed + 1000,
        )
    )

    print(
        "Patches de treino por época:",
        len(loaders["train"].sampler),
    )

    print(
        "Patches de validação por época:",
        len(val_loader.sampler),
    )

    # -----------------------------------------------------
    # Modelo
    # -----------------------------------------------------

    modelo = criar_resnet50(
        usar_pesos_pre_treinados=True,
        congelar_backbone_inicialmente=True,
    )

    modelo = modelo.to(
        dispositivo
    )

    treinaveis = [
        nome
        for nome, parametro
        in modelo.named_parameters()
        if parametro.requires_grad
    ]

    if not all(
        nome.startswith("fc.")
        for nome in treinaveis
    ):
        raise AssertionError(
            "Existem parâmetros treináveis "
            "fora da cabeça classificadora."
        )

    # -----------------------------------------------------
    # Loss / optimizer
    # -----------------------------------------------------

    criterio = (
        nn.CrossEntropyLoss()
    )

    otimizador = (
        torch.optim.AdamW(
            modelo.fc.parameters(),
            lr=config.learning_rate,
            weight_decay=(
                config.weight_decay
            ),
        )
    )

    # -----------------------------------------------------
    # Arquivos
    # -----------------------------------------------------

    history_path = Path(
        "outputs/resnet50/"
        "head_history.csv"
    )

    config_path = Path(
        "outputs/resnet50/"
        "head_config.json"
    )

    checkpoint_path = Path(
        "models/"
        "resnet50_head_best.pt"
    )

    registro_config = (
        criar_registro_configuracao(
            config,
            dispositivo,
        )
    )

    salvar_configuracao(
        registro_config,
        config_path,
    )

    # -----------------------------------------------------
    # Treinamento
    # -----------------------------------------------------

    historico = []

    melhor_val_loss = float(
        "inf"
    )

    melhor_epoca = None

    for epoch in range(
        1,
        config.epochs + 1,
    ):

        print()
        print(
            f"===== Época "
            f"{epoch}/{config.epochs} ====="
        )

        train_loss, train_acc = (
            treinar_epoca(
                modelo,
                loaders["train"],
                criterio,
                otimizador,
                dispositivo,
            )
        )

        val_loss, val_acc = (
            validar_epoca(
                modelo,
                val_loader,
                criterio,
                dispositivo,
            )
        )

        eh_melhor = (
            val_loss
            < melhor_val_loss
        )

        registro_epoca = {
            "epoch": epoch,
            "train_loss": (
                train_loss
            ),
            "train_accuracy": (
                train_acc
            ),
            "val_loss": (
                val_loss
            ),
            "val_accuracy": (
                val_acc
            ),
            "best": eh_melhor,
        }

        historico.append(
            registro_epoca
        )

        print(
            f"Train loss: {train_loss:.4f}"
        )
        print(
            f"Train accuracy: {train_acc:.4f}"
        )
        print(
            f"Validation loss: {val_loss:.4f}"
        )
        print(
            f"Validation accuracy: {val_acc:.4f}"
        )

        if eh_melhor:

            melhor_val_loss = (
                val_loss
            )

            melhor_epoca = (
                epoch
            )

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

            print(
                "Novo melhor checkpoint salvo."
            )

        salvar_historico(
            historico,
            history_path,
        )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    if not checkpoint_path.exists():
        raise AssertionError(
            "Nenhum checkpoint foi salvo."
        )

    print()
    print(
        "=== Treinamento da cabeça concluído ==="
    )

    print(
        "Melhor época:",
        melhor_epoca,
    )

    print(
        "Melhor validation loss:",
        f"{melhor_val_loss:.4f}",
    )

    print(
        "Histórico:",
        history_path,
    )

    print(
        "Configuração:",
        config_path,
    )

    print(
        "Melhor checkpoint:",
        checkpoint_path,
    )


# =========================================================
# CLI
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--train-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--val-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    config = HeadTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_patches_per_painting=(
            args.train_k
        ),
        val_patches_per_painting=(
            args.val_k
        ),
        num_workers=args.num_workers,
    )

    executar_treinamento(
        config
    )


if __name__ == "__main__":
    main()