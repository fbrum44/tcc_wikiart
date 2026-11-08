"""Etapa 8A - Definição do Vision Transformer (ViT-B/16) para classificação de estilos artísticos.

Espelha src/models/resnet.py: mesmos nomes de funções e mesma política de congelamento.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from src.data.labels import CLASSES_FINAIS
from src.data.normalization import HALF_MEAN, HALF_STD

NUM_CLASSES = len(CLASSES_FINAIS)

# ViT-B/16 pré-treinado no ImageNet-21k e ajustado no ImageNet-1k (AugReg).
VIT_MODEL_NAME = "vit_base_patch16_224.augreg_in21k_ft_in1k"
VIT_WEIGHTS = "augreg_in21k_ft_in1k"

# Mesmo patch 256x256 da ResNet. O timm interpola o embedding de posição (14x14 -> 16x16).
INPUT_SIZE = 256


def criar_vit(
    *,
    usar_pesos_pre_treinados: bool = True,
    congelar_backbone_inicialmente: bool = True,
) -> nn.Module:
    """
    Cria o ViT-B/16 utilizado no experimento.

    - pesos pré-treinados (augreg_in21k_ft_in1k);
    - cabeça (`head`) substituída por quatro classes;
    - backbone pode iniciar congelado para treinamento da cabeça.
    """

    modelo = timm.create_model(
        VIT_MODEL_NAME,
        pretrained=usar_pesos_pre_treinados,
        num_classes=NUM_CLASSES,
        img_size=INPUT_SIZE,
    )

    if congelar_backbone_inicialmente:
        congelar_backbone(modelo)

    return modelo


def congelar_backbone(modelo: nn.Module) -> None:
    """Congela todo o ViT, deixando treinável apenas a cabeça classificadora."""

    for parametro in modelo.parameters():
        parametro.requires_grad = False

    for parametro in modelo.head.parameters():
        parametro.requires_grad = True


def descongelar_modelo(modelo: nn.Module) -> None:
    """Libera todos os parâmetros para a etapa de fine-tuning."""

    for parametro in modelo.parameters():
        parametro.requires_grad = True


def contar_parametros(modelo: nn.Module) -> tuple[int, int]:
    """Retorna: total de parâmetros, parâmetros treináveis."""

    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, treinaveis


def validar_normalizacao() -> None:
    """
    Confirma que a normalização da Etapa 6 (preset "half", 0,5/0,5)
    coincide com a esperada pelos pesos escolhidos.
    """

    cfg = timm.create_model(VIT_MODEL_NAME, pretrained=False).pretrained_cfg

    mean_pesos = tuple(float(x) for x in cfg["mean"])
    std_pesos = tuple(float(x) for x in cfg["std"])

    if mean_pesos != tuple(HALF_MEAN):
        raise AssertionError(f"Mean incompatível: pesos={mean_pesos}, pipeline={HALF_MEAN}")

    if std_pesos != tuple(HALF_STD):
        raise AssertionError(f"STD incompatível: pesos={std_pesos}, pipeline={HALF_STD}")


def testar_vit(*, usar_pesos_pre_treinados: bool = True) -> None:
    """Smoke test estrutural da Etapa 8A."""

    print("Criando ViT-B/16...")

    modelo = criar_vit(
        usar_pesos_pre_treinados=usar_pesos_pre_treinados,
        congelar_backbone_inicialmente=True,
    )

    if NUM_CLASSES != 4:
        raise AssertionError(f"Esperadas 4 classes, encontradas {NUM_CLASSES}")

    if modelo.head.out_features != NUM_CLASSES:
        raise AssertionError("A camada final não possui quatro saídas.")

    validar_normalizacao()

    total, treinaveis = contar_parametros(modelo)
    parametros_head = sum(p.numel() for p in modelo.head.parameters())

    if treinaveis != parametros_head:
        raise AssertionError(
            "Durante a fase inicial somente a camada classificadora deveria estar treinável."
        )

    modelo.eval()
    entrada = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        saida = modelo(entrada)

    if saida.shape != (2, NUM_CLASSES):
        raise AssertionError(f"Saída inesperada: {saida.shape}")

    descongelar_modelo(modelo)
    _, treinaveis_finetuning = contar_parametros(modelo)

    if treinaveis_finetuning != total:
        raise AssertionError("Nem todos os parâmetros foram liberados para fine-tuning.")

    print()
    print("ViT-B/16 validado com sucesso.")
    print("Classes:", CLASSES_FINAIS)
    print("Saída:", tuple(saida.shape))
    print("Parâmetros totais:", f"{total:,}")
    print("Treináveis com backbone congelado:", f"{treinaveis:,}")
    print("Treináveis após descongelar:", f"{treinaveis_finetuning:,}")
    print("Normalização compatível:", "SIM")


if __name__ == "__main__":
    testar_vit()
