"""Augmentation dos patches de treino.

Regras:
- augmentation somente no split de treino;
- validação e teste permanecem inalterados;
- transformações conservadoras;
- tamanho final sempre 256x256;
- representação mantida em RGB.
"""

from __future__ import annotations

import cv2
import numpy as np


PATCH_SIZE = 256

HORIZONTAL_FLIP_PROB = 0.50
ROTATION_PROB = 0.50
MAX_ROTATION_DEGREES = 10.0


def validar_patch(
    patch: np.ndarray,
) -> None:
    """Verifica estrutura básica do patch."""

    if not isinstance(patch, np.ndarray):
        raise TypeError(
            "O patch deve ser um numpy.ndarray."
        )

    if patch.shape != (
        PATCH_SIZE,
        PATCH_SIZE,
        3,
    ):
        raise ValueError(
            "Dimensão inesperada do patch: "
            f"{patch.shape}. "
            f"Esperado: "
            f"({PATCH_SIZE}, {PATCH_SIZE}, 3)."
        )

    if patch.dtype != np.uint8:
        raise ValueError(
            "O patch deve estar em uint8. "
            f"Encontrado: {patch.dtype}"
        )


def aplicar_rotacao(
    patch: np.ndarray,
    angulo: float,
) -> np.ndarray:
    """
    Rotaciona o patch mantendo 256x256.

    BORDER_REFLECT_101 é utilizado para evitar
    regiões pretas artificiais nas extremidades.
    """

    centro = (
        (PATCH_SIZE - 1) / 2.0,
        (PATCH_SIZE - 1) / 2.0,
    )

    matriz = cv2.getRotationMatrix2D(
        centro,
        angulo,
        1.0,
    )

    rotacionado = cv2.warpAffine(
        patch,
        matriz,
        (
            PATCH_SIZE,
            PATCH_SIZE,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return rotacionado


def aplicar_augmentation(
    patch: np.ndarray,
    split: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Aplica augmentation somente ao conjunto de treino.

    Para validation e test, retorna uma cópia
    pixel a pixel do patch original.
    """

    validar_patch(
        patch
    )

    if split not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            f"Split inválido: {split}"
        )

    # Nunca modificar o array original.
    resultado = patch.copy()

    # Regra fundamental:
    # validação e teste não recebem augmentation.
    if split != "train":
        return resultado

    if rng is None:
        rng = np.random.default_rng()

    # -----------------------------------------------------
    # Flip horizontal
    # -----------------------------------------------------

    if rng.random() < HORIZONTAL_FLIP_PROB:
        resultado = cv2.flip(
            resultado,
            1,
        )

    # -----------------------------------------------------
    # Rotação pequena
    # -----------------------------------------------------

    if rng.random() < ROTATION_PROB:
        angulo = rng.uniform(
            -MAX_ROTATION_DEGREES,
            MAX_ROTATION_DEGREES,
        )

        resultado = aplicar_rotacao(
            resultado,
            angulo,
        )

    validar_patch(
        resultado
    )

    return resultado


def testar_augmentation() -> None:
    """Executa verificações simples do pipeline."""

    # Patch artificial com padrão assimétrico.
    patch = np.zeros(
        (
            PATCH_SIZE,
            PATCH_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    patch[
        20:100,
        30:80,
    ] = (
        255,
        100,
        50,
    )

    # -----------------------------------------------------
    # Validation não pode mudar.
    # -----------------------------------------------------

    validation = aplicar_augmentation(
        patch,
        "validation",
    )

    if not np.array_equal(
        patch,
        validation,
    ):
        raise AssertionError(
            "Validation foi alterado pelo augmentation."
        )

    # -----------------------------------------------------
    # Test não pode mudar.
    # -----------------------------------------------------

    test = aplicar_augmentation(
        patch,
        "test",
    )

    if not np.array_equal(
        patch,
        test,
    ):
        raise AssertionError(
            "Test foi alterado pelo augmentation."
        )

    # -----------------------------------------------------
    # Train deve preservar formato e dtype.
    # Seed fixa apenas para o teste.
    # -----------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    treino = aplicar_augmentation(
        patch,
        "train",
        rng=rng,
    )

    if treino.shape != patch.shape:
        raise AssertionError(
            "Augmentation alterou a dimensão."
        )

    if treino.dtype != patch.dtype:
        raise AssertionError(
            "Augmentation alterou o dtype."
        )

    if np.shares_memory(
        patch,
        treino,
    ):
        raise AssertionError(
            "Augmentation reutilizou "
            "a memória do patch original."
        )

    print(
        "Validation: inalterado."
    )

    print(
        "Test: inalterado."
    )

    print(
        "Train: augmentation válido."
    )

    print(
        "Shape preservado:",
        treino.shape,
    )

    print(
        "dtype preservado:",
        treino.dtype,
    )

    print(
        "Testes de augmentation "
        "concluídos com sucesso."
    )


if __name__ == "__main__":
    testar_augmentation()