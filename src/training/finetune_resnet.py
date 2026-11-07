"""Etapa 7 - Fine-tuning da ResNet-50."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchvision
from torch import nn

from src.data.labels import CLASSES_FINAIS
from src.data.normalization import IMAGENET_MEAN, IMAGENET_STD
from src.data.patch_dataset import (
    WikiArtImageSource,
    build_loaders,
)
from src.models.resnet import (
    RESNET50_WEIGHTS,
    criar_resnet50,
    descongelar_modelo,
)
from src.training.train_resnet import (
    configurar_seed,
    obter_dispositivo,
)
from src.training.train_resnet_head import (
    criar_loader_validacao_fixo,
)


# =========================================================
# Configuração
# =========================================================

@dataclass(frozen=True)
class FineTuneConfig:
    seed: int = 42

    epochs: int = 3
    batch_size: int = 32

    train_patches_per_painting: int = 8
    val_patches_per_painting: int = 8

    backbone_learning_rate: float = 1e-5
    fc_learning_rate: float = 1e-4

    weight_decay: float = 1e-4

    num_workers: int = 0

    architecture: str = "resnet50"
    weights: str = "IMAGENET1K_V2"

    input_size: int = 256

    source_checkpoint: str = (
        "models/resnet50_head_best.pt"
    )


# =========================================================
# BatchNorm
# =========================================================

def configurar_modo_finetuning(
    modelo: nn.Module,
) -> None:
    """
    Coloca o modelo em modo de treino, mas mantém
    as estatísticas das BatchNorm congeladas.

    Os parâmetros affine (weight/bias) das BatchNorm
    continuam treináveis.
    """

    modelo.train()

    for modulo in modelo.modules():
        if isinstance(
            modulo,
            nn.BatchNorm2d,
        ):
            modulo.eval()


# =========================================================
# Carregamento do checkpoint da cabeça
# =========================================================

def carregar_modelo_da_cabeca(
    checkpoint_path: Path,
) -> tuple[nn.Module, dict]:

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    classes_checkpoint = (
        checkpoint.get("classes")
    )

    if (
        list(classes_checkpoint)
        != list(CLASSES_FINAIS)
    ):
        raise ValueError(
            "As classes do checkpoint não "
            "coincidem com CLASSES_FINAIS."
        )

    modelo = criar_resnet50(
        usar_pesos_pre_treinados=False,
        congelar_backbone_inicialmente=False,
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    descongelar_modelo(
        modelo
    )

    return modelo, checkpoint


# =========================================================
# Otimizador
# =========================================================

def criar_otimizador(
    modelo: nn.Module,
    config: FineTuneConfig,
):
    """
    Backbone e cabeça recebem learning rates diferentes.
    """

    parametros_backbone = [
        parametro
        for nome, parametro
        in modelo.named_parameters()
        if not nome.startswith("fc.")
    ]

    parametros_fc = list(
        modelo.fc.parameters()
    )

    otimizador = torch.optim.AdamW(
        [
            {
                "params": parametros_backbone,
                "lr": (
                    config.backbone_learning_rate
                ),
            },
            {
                "params": parametros_fc,
                "lr": (
                    config.fc_learning_rate
                ),
            },
        ],
        weight_decay=(
            config.weight_decay
        ),
    )

    return otimizador


# =========================================================
# Treino
# =========================================================

def treinar_epoca(
    modelo: nn.Module,
    loader,
    criterio: nn.Module,
    otimizador,
    dispositivo: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[float, float]:

    configurar_modo_finetuning(
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

        if (
            max_batches is not None
            and batch_idx > max_batches
        ):
            break

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
                f"Loss inválida: "
                f"{loss.item()}"
            )

        loss.backward()

        otimizador.step()

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

        limite_exibido = (
            min(total_batches, max_batches)
            if max_batches is not None
            else total_batches
        )

        if (
            batch_idx == 1
            or batch_idx % 25 == 0
            or batch_idx == limite_exibido
        ):
            print(
                f"  treino: "
                f"{batch_idx}/"
                f"{limite_exibido} batches"
            )

    if total == 0:
        raise RuntimeError(
            "Nenhuma amostra processada."
        )

    return (
        soma_loss / total,
        corretos / total,
    )


# =========================================================
# Validação
# =========================================================

def validar_epoca(
    modelo: nn.Module,
    loader,
    criterio: nn.Module,
    dispositivo: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[float, float]:

    modelo.eval()

    soma_loss = 0.0
    corretos = 0
    total = 0

    with torch.no_grad():

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
                    "Loss de validação "
                    f"inválida: {loss.item()}"
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
        "finetune_epoch",
        "overall_epoch",
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
# Registro de configuração
# =========================================================

def criar_registro_configuracao(
    config: FineTuneConfig,
    dispositivo: torch.device,
    checkpoint_cabeca: dict,
) -> dict:

    registro = asdict(
        config
    )

    registro.update(
        {
            "classes": (
                CLASSES_FINAIS
            ),
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
            "phase": (
                "full_finetuning"
            ),
            "batchnorm_running_stats": (
                "frozen"
            ),
            "validation_strategy": (
                "fixed_patches_per_painting"
            ),
            "checkpoint_metric": (
                "validation_loss"
            ),
            "source_head_epoch": (
                checkpoint_cabeca.get(
                    "epoch"
                )
            ),
            "source_head_val_loss": (
                checkpoint_cabeca.get(
                    "val_loss"
                )
            ),
            "source_head_val_accuracy": (
                checkpoint_cabeca.get(
                    "val_accuracy"
                )
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

def salvar_checkpoint(
    caminho: Path,
    modelo: nn.Module,
    otimizador,
    *,
    finetune_epoch: int,
    overall_epoch: int,
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
            "phase": (
                "full_finetuning"
            ),
            "finetune_epoch": (
                finetune_epoch
            ),
            "overall_epoch": (
                overall_epoch
            ),
            "classes": (
                CLASSES_FINAIS
            ),
            "model_state_dict": (
                modelo.state_dict()
            ),
            "optimizer_state_dict": (
                otimizador.state_dict()
            ),
            "train_loss": (
                train_loss
            ),
            "train_accuracy": (
                train_accuracy
            ),
            "val_loss": (
                val_loss
            ),
            "val_accuracy": (
                val_accuracy
            ),
            "config": config,
            "history": historico,
        },
        caminho,
    )


# =========================================================
# Experimento
# =========================================================

def executar_finetuning(
    config: FineTuneConfig,
    *,
    smoke: bool = False,
) -> None:

    print(
        "=== Etapa 7 — Fine-tuning ResNet-50 ==="
    )

    configurar_seed(
        config.seed
    )

    dispositivo = (
        obter_dispositivo()
    )

    print(
        "Dispositivo:",
        dispositivo,
    )

    # -----------------------------------------------------
    # Checkpoint da cabeça
    # -----------------------------------------------------

    source_path = Path(
        config.source_checkpoint
    )

    modelo, checkpoint_cabeca = (
        carregar_modelo_da_cabeca(
            source_path
        )
    )

    print(
        "Checkpoint inicial:",
        source_path,
    )

    print(
        "Validation loss da cabeça:",
        checkpoint_cabeca.get(
            "val_loss"
        ),
    )

    print(
        "Validation accuracy da cabeça:",
        checkpoint_cabeca.get(
            "val_accuracy"
        ),
    )

    modelo = modelo.to(
        dispositivo
    )

    # Todos devem estar treináveis agora.
    nao_treinaveis = [
        nome
        for nome, parametro
        in modelo.named_parameters()
        if not parametro.requires_grad
    ]

    if nao_treinaveis:
        raise AssertionError(
            "Existem parâmetros congelados "
            f"no fine-tuning: {nao_treinaveis[:5]}"
        )

    # -----------------------------------------------------
    # Snapshots para validação posterior
    # -----------------------------------------------------

    conv1_antes = (
        modelo.conv1.weight
        .detach()
        .cpu()
        .clone()
    )

    fc_antes = (
        modelo.fc.weight
        .detach()
        .cpu()
        .clone()
    )

    bn_media_antes = (
        modelo.bn1.running_mean
        .detach()
        .cpu()
        .clone()
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
    # Loss / otimizador
    # -----------------------------------------------------

    criterio = (
        nn.CrossEntropyLoss()
    )

    otimizador = (
        criar_otimizador(
            modelo,
            config,
        )
    )

    print(
        "LR backbone:",
        config.backbone_learning_rate,
    )

    print(
        "LR fc:",
        config.fc_learning_rate,
    )

    # -----------------------------------------------------
    # Saídas
    # -----------------------------------------------------

    if smoke:
        history_path = Path(
            "outputs/resnet50/"
            "finetune_smoke_history.csv"
        )

        config_path = Path(
            "outputs/resnet50/"
            "finetune_smoke_config.json"
        )

        checkpoint_path = Path(
            "models/"
            "resnet50_finetune_smoke.pt"
        )

    else:
        history_path = Path(
            "outputs/resnet50/"
            "finetune_history.csv"
        )

        config_path = Path(
            "outputs/resnet50/"
            "finetune_config.json"
        )

        checkpoint_path = Path(
            "models/"
            "resnet50_finetune_best.pt"
        )

    registro_config = (
        criar_registro_configuracao(
            config,
            dispositivo,
            checkpoint_cabeca,
        )
    )

    registro_config["smoke_test"] = (
        smoke
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

    epochs_executar = (
        1
        if smoke
        else config.epochs
    )

    head_epoch = int(
        checkpoint_cabeca.get(
            "epoch",
            3,
        )
    )

    for finetune_epoch in range(
        1,
        epochs_executar + 1,
    ):

        overall_epoch = (
            head_epoch
            + finetune_epoch
        )

        print()
        print(
            f"===== Fine-tuning "
            f"{finetune_epoch}/"
            f"{epochs_executar} "
            f"(época geral "
            f"{overall_epoch}) ====="
        )

        train_loss, train_acc = (
            treinar_epoca(
                modelo,
                loaders["train"],
                criterio,
                otimizador,
                dispositivo,
                max_batches=(
                    5
                    if smoke
                    else None
                ),
            )
        )

        val_loss, val_acc = (
            validar_epoca(
                modelo,
                val_loader,
                criterio,
                dispositivo,
                max_batches=(
                    3
                    if smoke
                    else None
                ),
            )
        )

        eh_melhor = (
            val_loss
            < melhor_val_loss
        )

        registro_epoca = {
            "finetune_epoch": (
                finetune_epoch
            ),
            "overall_epoch": (
                overall_epoch
            ),
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
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_acc:.4f}"
        )

        print(
            f"Validation loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{val_acc:.4f}"
        )

        if eh_melhor:

            melhor_val_loss = (
                val_loss
            )

            melhor_epoca = (
                finetune_epoch
            )

            salvar_checkpoint(
                checkpoint_path,
                modelo,
                otimizador,
                finetune_epoch=(
                    finetune_epoch
                ),
                overall_epoch=(
                    overall_epoch
                ),
                train_loss=(
                    train_loss
                ),
                train_accuracy=(
                    train_acc
                ),
                val_loss=(
                    val_loss
                ),
                val_accuracy=(
                    val_acc
                ),
                config=(
                    registro_config
                ),
                historico=(
                    historico
                ),
            )

            print(
                "Novo melhor checkpoint "
                "do fine-tuning salvo."
            )

        salvar_historico(
            historico,
            history_path,
        )

    # -----------------------------------------------------
    # Validação estrutural
    # -----------------------------------------------------

    conv1_depois = (
        modelo.conv1.weight
        .detach()
        .cpu()
    )

    fc_depois = (
        modelo.fc.weight
        .detach()
        .cpu()
    )

    bn_media_depois = (
        modelo.bn1.running_mean
        .detach()
        .cpu()
    )

    if torch.equal(
        conv1_antes,
        conv1_depois,
    ):
        raise AssertionError(
            "O backbone não foi atualizado "
            "durante o fine-tuning."
        )

    if torch.equal(
        fc_antes,
        fc_depois,
    ):
        raise AssertionError(
            "A fc não foi atualizada "
            "durante o fine-tuning."
        )

    if not torch.equal(
        bn_media_antes,
        bn_media_depois,
    ):
        raise AssertionError(
            "As estatísticas BatchNorm "
            "foram alteradas."
        )

    if not checkpoint_path.exists():
        raise AssertionError(
            "Checkpoint não foi criado."
        )

    # -----------------------------------------------------
    # Comparação com a cabeça
    # -----------------------------------------------------

    head_val_loss = float(
        checkpoint_cabeca[
            "val_loss"
        ]
    )

    print()
    print(
        "=== Fine-tuning concluído ==="
    )

    print(
        "Melhor época do fine-tuning:",
        melhor_epoca,
    )

    print(
        "Melhor validation loss "
        "do fine-tuning:",
        f"{melhor_val_loss:.4f}",
    )

    print(
        "Validation loss da cabeça:",
        f"{head_val_loss:.4f}",
    )

    if melhor_val_loss < head_val_loss:
        print(
            "Fine-tuning superou "
            "o checkpoint da cabeça: SIM"
        )
    else:
        print(
            "Fine-tuning superou "
            "o checkpoint da cabeça: NÃO"
        )

    print(
        "Backbone foi atualizado: SIM"
    )

    print(
        "FC foi atualizada: SIM"
    )

    print(
        "Estatísticas BatchNorm "
        "permaneceram congeladas: SIM"
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

    parser = (
        argparse.ArgumentParser()
    )

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

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    config = FineTuneConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_patches_per_painting=(
            args.train_k
        ),
        val_patches_per_painting=(
            args.val_k
        ),
        num_workers=(
            args.num_workers
        ),
    )

    executar_finetuning(
        config,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()