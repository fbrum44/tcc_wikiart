"""Etapa 7B - Smoke test do treinamento da cabeça classificadora da ResNet-50."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.labels import CLASSES_FINAIS
from src.data.patch_dataset import (
    WikiArtImageSource,
    build_loaders,
)
from src.models.resnet import (
    criar_resnet50,
)


SEED = 42

# Configuração propositalmente pequena:
# serve apenas para validar o pipeline.
SMOKE_BATCH_SIZE = 8
SMOKE_PATCHES_PER_PAINTING = 2

SMOKE_TRAIN_BATCHES = 5
SMOKE_VAL_BATCHES = 3

SMOKE_LR = 1e-3
SMOKE_WEIGHT_DECAY = 1e-4

CHECKPOINT_PATH = Path(
    "models/resnet50_smoke.pt"
)


# ---------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------

def configurar_seed(
    seed: int = SEED,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Favorece determinismo.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Dispositivo
# ---------------------------------------------------------

def obter_dispositivo() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ---------------------------------------------------------
# Modo da fase 1
# ---------------------------------------------------------

def configurar_modo_cabeca(
    modelo: nn.Module,
) -> None:
    """
    Backbone permanece em eval para impedir atualização
    das estatísticas BatchNorm.

    A cabeça fc permanece disponível para treinamento.
    """

    modelo.eval()

    # Linear não possui comportamento diferente entre
    # train/eval, mas deixamos explícita a intenção.
    modelo.fc.train()


# ---------------------------------------------------------
# Treino
# ---------------------------------------------------------

def treinar_cabeca(
    modelo: nn.Module,
    loader,
    criterio: nn.Module,
    otimizador,
    dispositivo: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[float, float]:
    configurar_modo_cabeca(
        modelo
    )

    soma_loss = 0.0
    corretos = 0
    total = 0

    batches_processados = 0

    for batch_idx, batch in enumerate(loader):

        if (
            max_batches is not None
            and batch_idx >= max_batches
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
                f"Loss inválida: {loss.item()}"
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

        batches_processados += 1

        print(
            f"Treino batch "
            f"{batches_processados}: "
            f"loss={loss.item():.4f}"
        )

    if total == 0:
        raise RuntimeError(
            "Nenhum exemplo foi processado no treino."
        )

    loss_media = (
        soma_loss / total
    )

    accuracy = (
        corretos / total
    )

    return (
        loss_media,
        accuracy,
    )


# ---------------------------------------------------------
# Validação
# ---------------------------------------------------------

def validar(
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

        for batch_idx, batch in enumerate(loader):

            if (
                max_batches is not None
                and batch_idx >= max_batches
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
                    f"Loss de validação inválida: "
                    f"{loss.item()}"
                )

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

    if total == 0:
        raise RuntimeError(
            "Nenhum exemplo foi processado "
            "na validação."
        )

    return (
        soma_loss / total,
        corretos / total,
    )


# ---------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------

def salvar_checkpoint(
    caminho: Path,
    modelo: nn.Module,
    otimizador,
    *,
    train_loss: float,
    val_loss: float,
    train_accuracy: float,
    val_accuracy: float,
) -> None:

    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "fase": "smoke_head_only",
        "seed": SEED,
        "classes": CLASSES_FINAIS,
        "model_state_dict": (
            modelo.state_dict()
        ),
        "optimizer_state_dict": (
            otimizador.state_dict()
        ),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "learning_rate": SMOKE_LR,
        "weight_decay": (
            SMOKE_WEIGHT_DECAY
        ),
        "batch_size": (
            SMOKE_BATCH_SIZE
        ),
        "patches_per_painting": (
            SMOKE_PATCHES_PER_PAINTING
        ),
        "torch_version": (
            torch.__version__
        ),
    }

    torch.save(
        checkpoint,
        caminho,
    )


# ---------------------------------------------------------
# Smoke test
# ---------------------------------------------------------

def executar_smoke_test() -> None:

    print(
        "=== Etapa 7B — Smoke test ResNet-50 ==="
    )

    configurar_seed()

    dispositivo = obter_dispositivo()

    print(
        "Dispositivo:",
        dispositivo,
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    image_source = WikiArtImageSource()

    loaders = build_loaders(
        manifest_dir="data/metadata",
        image_source=image_source,
        use_augmentation=True,
        batch_size=SMOKE_BATCH_SIZE,
        patches_per_painting=(
            SMOKE_PATCHES_PER_PAINTING
        ),
        num_workers=0,
        seed=SEED,
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

    # -----------------------------------------------------
    # Garantir que somente fc treina
    # -----------------------------------------------------

    parametros_treinaveis = [
        nome
        for nome, parametro
        in modelo.named_parameters()
        if parametro.requires_grad
    ]

    if not all(
        nome.startswith("fc.")
        for nome in parametros_treinaveis
    ):
        raise AssertionError(
            "Foram encontrados parâmetros "
            "treináveis fora da fc."
        )

    print(
        "Parâmetros treináveis:",
        parametros_treinaveis,
    )

    # -----------------------------------------------------
    # Snapshots anteriores
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
    # Loss / optimizer
    # -----------------------------------------------------

    criterio = nn.CrossEntropyLoss()

    otimizador = torch.optim.AdamW(
        modelo.fc.parameters(),
        lr=SMOKE_LR,
        weight_decay=SMOKE_WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Treino curto
    # -----------------------------------------------------

    train_loss, train_accuracy = (
        treinar_cabeca(
            modelo,
            loaders["train"],
            criterio,
            otimizador,
            dispositivo,
            max_batches=(
                SMOKE_TRAIN_BATCHES
            ),
        )
    )

    # -----------------------------------------------------
    # Validação curta
    # -----------------------------------------------------

    val_loss, val_accuracy = validar(
        modelo,
        loaders["validation"],
        criterio,
        dispositivo,
        max_batches=(
            SMOKE_VAL_BATCHES
        ),
    )

    # -----------------------------------------------------
    # Verificações após otimização
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

    # Backbone não pode mudar.
    if not torch.equal(
        conv1_antes,
        conv1_depois,
    ):
        raise AssertionError(
            "O backbone foi alterado "
            "durante o treino da cabeça."
        )

    # FC precisa mudar.
    if torch.equal(
        fc_antes,
        fc_depois,
    ):
        raise AssertionError(
            "A camada fc não foi atualizada."
        )

    # BatchNorm do backbone também deve permanecer intacta.
    if not torch.equal(
        bn_media_antes,
        bn_media_depois,
    ):
        raise AssertionError(
            "As estatísticas BatchNorm "
            "do backbone foram alteradas."
        )

    # -----------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------

    salvar_checkpoint(
        CHECKPOINT_PATH,
        modelo,
        otimizador,
        train_loss=train_loss,
        val_loss=val_loss,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
    )

    if not CHECKPOINT_PATH.exists():
        raise AssertionError(
            "Checkpoint não foi criado."
        )

    # Teste básico de leitura.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint["classes"]
        != CLASSES_FINAIS
    ):
        raise AssertionError(
            "Classes incorretas no checkpoint."
        )

    # -----------------------------------------------------
    # Resultado
    # -----------------------------------------------------

    print()
    print(
        "Smoke test concluído com sucesso."
    )

    print(
        f"Train loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        "Backbone permaneceu congelado: SIM"
    )

    print(
        "BatchNorm permaneceu congelada: SIM"
    )

    print(
        "Cabeça classificadora foi atualizada: SIM"
    )

    print(
        "Checkpoint criado:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    executar_smoke_test()