"""Etapa 7A - Definição da ResNet-50 para classificação de estilos artísticos."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from src.data.labels import CLASSES_FINAIS
from src.data.normalization import IMAGENET_MEAN, IMAGENET_STD


NUM_CLASSES = len(CLASSES_FINAIS)

RESNET50_WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


def criar_resnet50(
    *,
    usar_pesos_pre_treinados: bool = True,
    congelar_backbone_inicialmente: bool = True,
) -> nn.Module:
    """
    Cria a ResNet-50 utilizada no experimento.

    - pesos ImageNet1K V2;
    - camada final substituída por quatro classes;
    - backbone pode iniciar congelado para treinamento da cabeça.
    """

    weights = (
        RESNET50_WEIGHTS
        if usar_pesos_pre_treinados
        else None
    )

    modelo = resnet50(
        weights=weights
    )

    # A ResNet-50 entrega 2048 features para a camada classificadora.
    in_features = modelo.fc.in_features

    modelo.fc = nn.Linear(
        in_features,
        NUM_CLASSES,
    )

    if congelar_backbone_inicialmente:
        congelar_backbone(modelo)

    return modelo


def congelar_backbone(
    modelo: nn.Module,
) -> None:
    """
    Congela toda a ResNet, deixando treinável apenas a cabeça classificadora.
    """

    for parametro in modelo.parameters():
        parametro.requires_grad = False

    for parametro in modelo.fc.parameters():
        parametro.requires_grad = True


def descongelar_modelo(
    modelo: nn.Module,
) -> None:
    """
    Libera todos os parâmetros para a etapa posterior de fine-tuning.
    """

    for parametro in modelo.parameters():
        parametro.requires_grad = True


def contar_parametros(
    modelo: nn.Module,
) -> tuple[int, int]:
    """
    Retorna:
        total de parâmetros,
        parâmetros treináveis.
    """

    total = sum(
        p.numel()
        for p in modelo.parameters()
    )

    treinaveis = sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )

    return total, treinaveis


def validar_normalizacao() -> None:
    """
    Confirma que a normalização da Etapa 6 coincide
    com a esperada pelos pesos escolhidos.
    """

    transforms = RESNET50_WEIGHTS.transforms()

    mean_pesos = tuple(
        float(x)
        for x in transforms.mean
    )

    std_pesos = tuple(
        float(x)
        for x in transforms.std
    )

    if mean_pesos != tuple(IMAGENET_MEAN):
        raise AssertionError(
            f"Mean incompatível: pesos={mean_pesos}, "
            f"pipeline={IMAGENET_MEAN}"
        )

    if std_pesos != tuple(IMAGENET_STD):
        raise AssertionError(
            f"STD incompatível: pesos={std_pesos}, "
            f"pipeline={IMAGENET_STD}"
        )


def testar_resnet50() -> None:
    """
    Smoke test estrutural da Etapa 7A.
    """

    print("Criando ResNet-50...")

    modelo = criar_resnet50(
        usar_pesos_pre_treinados=True,
        congelar_backbone_inicialmente=True,
    )

    # -----------------------------------------------------
    # Classes
    # -----------------------------------------------------

    if NUM_CLASSES != 4:
        raise AssertionError(
            f"Esperadas 4 classes, encontradas {NUM_CLASSES}"
        )

    if modelo.fc.out_features != NUM_CLASSES:
        raise AssertionError(
            "A camada final não possui quatro saídas."
        )

    # -----------------------------------------------------
    # Normalização
    # -----------------------------------------------------

    validar_normalizacao()

    # -----------------------------------------------------
    # Cabeça congelada / treinável
    # -----------------------------------------------------

    total, treinaveis = contar_parametros(
        modelo
    )

    parametros_fc = sum(
        p.numel()
        for p in modelo.fc.parameters()
    )

    if treinaveis != parametros_fc:
        raise AssertionError(
            "Durante a fase inicial somente a camada "
            "classificadora deveria estar treinável."
        )

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    modelo.eval()

    entrada = torch.randn(
        2,
        3,
        256,
        256,
    )

    with torch.no_grad():
        saida = modelo(
            entrada
        )

    if saida.shape != (
        2,
        NUM_CLASSES,
    ):
        raise AssertionError(
            f"Saída inesperada: {saida.shape}"
        )

    # -----------------------------------------------------
    # Fine-tuning
    # -----------------------------------------------------

    descongelar_modelo(
        modelo
    )

    _, treinaveis_finetuning = contar_parametros(
        modelo
    )

    if treinaveis_finetuning != total:
        raise AssertionError(
            "Nem todos os parâmetros foram liberados "
            "para fine-tuning."
        )

    # -----------------------------------------------------
    # Resultado
    # -----------------------------------------------------

    print()
    print("ResNet-50 validada com sucesso.")
    print(
        "Classes:",
        CLASSES_FINAIS,
    )
    print(
        "Saída:",
        tuple(saida.shape),
    )
    print(
        "Parâmetros totais:",
        f"{total:,}",
    )
    print(
        "Treináveis com backbone congelado:",
        f"{treinaveis:,}",
    )
    print(
        "Treináveis após descongelar:",
        f"{treinaveis_finetuning:,}",
    )
    print(
        "Normalização compatível:",
        "SIM",
    )


if __name__ == "__main__":
    testar_resnet50()