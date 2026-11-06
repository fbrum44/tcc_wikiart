"""Etapa 6 - Normalização dos patches compatível com os pesos pré-treinados.

Os patches chegam como arrays uint8 RGB (256, 256, 3) e saem como float32 CHW.

ATENÇÃO: a normalização precisa ser a MESMA usada no pré-treino do modelo.
- ResNet com pesos do ImageNet (torchvision/timm): preset "imagenet".
- Vision Transformer: depende do checkpoint. Alguns usam média/desvio 0,5
  (preset "half"). Confirme na documentação do checkpoint escolhido.
"""
from __future__ import annotations

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)

PRESETS = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "half": (HALF_MEAN, HALF_STD),
}


def get_stats(name: str = "imagenet") -> tuple[tuple, tuple]:
    if name not in PRESETS:
        raise ValueError(f"Preset desconhecido: {name!r}. Opções: {list(PRESETS)}")
    return PRESETS[name]


def normalize_patch(patch: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """uint8 HWC RGB -> float32 CHW, (x/255 - mean) / std."""
    if patch.dtype != np.uint8 or patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"Esperado uint8 (H, W, 3); recebido {patch.dtype} {patch.shape}")
    x = patch.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def denormalize_patch(chw: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """float32 CHW normalizado -> uint8 HWC (para visualização)."""
    x = chw.transpose(1, 2, 0) * np.asarray(std, np.float32) + np.asarray(mean, np.float32)
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
