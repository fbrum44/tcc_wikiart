"""Etapa 5 - Geração determinística da grade e dos manifestos de patches.

Decisões metodológicas:
- Patch: 256x256 pixels.
- Stride nominal: 128 pixels.
- Sobreposição nominal: 50%.
- Nenhum resize.
- Nenhum padding.
- Nenhum descarte das bordas.
- O último patch é ajustado para terminar exatamente na borda.
- Se esse ajuste criaria um patch quase duplicado, o último patch regular
  é reposicionado em vez de adicionar um novo.
- Todos os patches preservam associação com Painting_ID e split.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
import cv2
import numpy as np

import pandas as pd


PATCH_SIZE = 256
STRIDE = 128

# Se o deslocamento necessário para encaixar a borda for menor
# do que metade do stride, reposicionamos o último patch em vez
# de adicionar um patch quase duplicado.
EDGE_REPOSITION_THRESHOLD = STRIDE // 2

SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}

EXPECTED_PAINTINGS = {
    "train": 7000,
    "validation": 1500,
    "test": 1500,
}

# Valores esperados para os manifestos congelados da Etapa 4
# usando PATCH_SIZE=256, STRIDE=128 e a política de bordas atual.
EXPECTED_PATCHES = {
    "train": 980_730,
    "validation": 209_730,
    "test": 209_030,
}

COLUNAS_OBRIGATORIAS = {
    "Painting_ID",
    "Indice_Dataset",
    "Macro_Estilo",
    "Estilo",
    "Largura",
    "Altura",
}

PATCH_COLUMNS = [
    "Patch_ID",
    "Painting_ID",
    "Indice_Dataset",
    "Split",
    "Macro_Estilo",
    "Estilo",
    "x0",
    "y0",
    "x1",
    "y1",
]

SUMMARY_COLUMNS = [
    "Painting_ID",
    "Indice_Dataset",
    "Split",
    "Macro_Estilo",
    "Estilo",
    "Largura",
    "Altura",
    "N_X",
    "N_Y",
    "N_Patches",
]


def require(condition: bool, message: str) -> None:
    """Interrompe a execução quando uma validação falha."""
    if not condition:
        raise AssertionError(message)


def calcular_posicoes(
    tamanho: int,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> list[int]:
    """
    Calcula as posições iniciais dos patches em um único eixo.

    A grade segue o stride nominal até a região final.

    Caso a última posição regular não cubra exatamente a borda:
    - se o deslocamento restante for pequeno, reposiciona o último patch;
    - caso contrário, adiciona um patch alinhado à borda.

    Isso garante cobertura completa sem padding e reduz patches
    quase duplicados nas extremidades.
    """

    if tamanho < patch_size:
        raise ValueError(
            f"Dimensão {tamanho} menor que PATCH_SIZE={patch_size}."
        )

    if stride <= 0:
        raise ValueError(
            "STRIDE deve ser positivo."
        )

    if stride > patch_size:
        raise ValueError(
            "STRIDE maior que PATCH_SIZE criaria regiões sem cobertura."
        )

    ultimo_inicio = tamanho - patch_size

    posicoes = list(
        range(
            0,
            ultimo_inicio + 1,
            stride,
        )
    )

    # Caso já termine exatamente na borda.
    if posicoes[-1] == ultimo_inicio:
        return posicoes

    deslocamento_final = (
        ultimo_inicio - posicoes[-1]
    )

    if (
        len(posicoes) >= 2
        and deslocamento_final < EDGE_REPOSITION_THRESHOLD
    ):
        # Evita adicionar um patch quase igual ao anterior.
        posicoes[-1] = ultimo_inicio

    else:
        # O deslocamento é suficientemente grande para justificar
        # um novo patch alinhado à borda.
        posicoes.append(
            ultimo_inicio
        )

    return posicoes


def validar_posicoes(
    posicoes: list[int],
    tamanho: int,
    patch_size: int = PATCH_SIZE,
) -> None:
    """Valida a cobertura de um eixo."""

    require(
        len(posicoes) > 0,
        "Nenhuma posição foi gerada.",
    )

    require(
        posicoes[0] == 0,
        "A grade não começa no início da imagem.",
    )

    require(
        posicoes[-1] + patch_size == tamanho,
        "A grade não termina exatamente na borda da imagem.",
    )

    require(
        posicoes == sorted(set(posicoes)),
        "Existem posições duplicadas ou fora de ordem.",
    )

    for inicio in posicoes:
        require(
            0 <= inicio <= tamanho - patch_size,
            f"Posição inválida: {inicio}",
        )

    # Como o patch possui 256 px, qualquer distância entre
    # inícios <= 256 mantém a cobertura contínua.
    for anterior, atual in zip(
        posicoes,
        posicoes[1:],
    ):
        distancia = atual - anterior

        require(
            distancia <= patch_size,
            (
                "Foi encontrada uma região sem cobertura: "
                f"distância={distancia}"
            ),
        )

        # Com a política adotada, não permitimos um novo patch
        # deslocado menos de 64 px do anterior.
        require(
            distancia >= EDGE_REPOSITION_THRESHOLD,
            (
                "Patches excessivamente redundantes na borda: "
                f"distância={distancia}"
            ),
        )


def gerar_grade(
    largura: int,
    altura: int,
) -> tuple[list[int], list[int]]:
    """Gera e valida as posições X e Y de uma pintura."""

    xs = calcular_posicoes(
        largura
    )

    ys = calcular_posicoes(
        altura
    )

    validar_posicoes(
        xs,
        largura,
    )

    validar_posicoes(
        ys,
        altura,
    )

    return xs, ys


def criar_patch_id(
    painting_id: str,
    x0: int,
    y0: int,
) -> str:
    """Cria identificador estável e legível para um patch."""

    return (
        f"{painting_id}"
        f"_x{x0:05d}"
        f"_y{y0:05d}"
    )


def validar_manifesto_pinturas(
    df: pd.DataFrame,
    split: str,
) -> None:
    """Valida o manifesto da Etapa 4 antes de gerar patches."""

    faltantes = (
        COLUNAS_OBRIGATORIAS
        - set(df.columns)
    )

    require(
        not faltantes,
        (
            f"{split}: colunas ausentes: "
            f"{sorted(faltantes)}"
        ),
    )

    require(
        df["Painting_ID"].is_unique,
        f"{split}: Painting_ID duplicado.",
    )

    require(
        df["Indice_Dataset"].is_unique,
        f"{split}: Indice_Dataset duplicado.",
    )

    require(
        len(df) == EXPECTED_PAINTINGS[split],
        (
            f"{split}: quantidade inesperada de pinturas. "
            f"Encontrado={len(df)}, "
            f"esperado={EXPECTED_PAINTINGS[split]}"
        ),
    )

    require(
        (df["Largura"] >= PATCH_SIZE).all(),
        f"{split}: existe pintura estreita demais.",
    )

    require(
        (df["Altura"] >= PATCH_SIZE).all(),
        f"{split}: existe pintura baixa demais.",
    )


def gerar_manifesto_split(
    df: pd.DataFrame,
    split: str,
    caminho_saida: Path,
) -> tuple[list[dict], int]:
    """
    Gera o manifesto completo de patches em CSV.GZ.

    O arquivo é escrito de forma incremental para evitar manter
    aproximadamente um milhão de registros em memória.
    """

    validar_manifesto_pinturas(
        df,
        split,
    )

    resumo = []

    total_patches = 0

    with gzip.open(
        caminho_saida,
        mode="wt",
        encoding="utf-8",
        newline="",
    ) as arquivo:

        writer = csv.DictWriter(
            arquivo,
            fieldnames=PATCH_COLUMNS,
        )

        writer.writeheader()

        for pintura in df.itertuples(
            index=False
        ):

            largura = int(
                pintura.Largura
            )

            altura = int(
                pintura.Altura
            )

            xs, ys = gerar_grade(
                largura,
                altura,
            )

            quantidade = (
                len(xs) * len(ys)
            )

            resumo.append({
                "Painting_ID": pintura.Painting_ID,
                "Indice_Dataset": pintura.Indice_Dataset,
                "Split": split,
                "Macro_Estilo": pintura.Macro_Estilo,
                "Estilo": pintura.Estilo,
                "Largura": largura,
                "Altura": altura,
                "N_X": len(xs),
                "N_Y": len(ys),
                "N_Patches": quantidade,
            })

            # Ordem determinística:
            # percorre primeiro Y e depois X.
            for y0 in ys:
                for x0 in xs:

                    x1 = x0 + PATCH_SIZE
                    y1 = y0 + PATCH_SIZE

                    patch_id = criar_patch_id(
                        pintura.Painting_ID,
                        x0,
                        y0,
                    )

                    writer.writerow({
                        "Patch_ID": patch_id,
                        "Painting_ID": pintura.Painting_ID,
                        "Indice_Dataset": pintura.Indice_Dataset,
                        "Split": split,
                        "Macro_Estilo": pintura.Macro_Estilo,
                        "Estilo": pintura.Estilo,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                    })

                    total_patches += 1

    require(
        total_patches == EXPECTED_PATCHES[split],
        (
            f"{split}: quantidade de patches inesperada. "
            f"Encontrado={total_patches:,}, "
            f"esperado={EXPECTED_PATCHES[split]:,}"
        ),
    )

    return resumo, total_patches

def validar_manifesto_patches(
    caminho_manifesto: Path,
    df_pinturas: pd.DataFrame,
    split: str,
) -> None:
    """
    Relê o manifesto de patches persistido e verifica
    integridade, coordenadas, associação à pintura original
    e ausência de inconsistências.
    """

    print(
        f"Validando manifesto persistido de {split}..."
    )

    patches = pd.read_csv(
        caminho_manifesto
    )

    esperado = EXPECTED_PATCHES[split]

    # ---------------------------------------------------------
    # Quantidade total
    # ---------------------------------------------------------

    require(
        len(patches) == esperado,
        (
            f"{split}: número incorreto de patches no arquivo. "
            f"Encontrado={len(patches):,}, "
            f"esperado={esperado:,}"
        ),
    )

    # ---------------------------------------------------------
    # Patch_ID
    # ---------------------------------------------------------

    require(
        patches["Patch_ID"].notna().all(),
        f"{split}: Patch_ID ausente.",
    )

    require(
        patches["Patch_ID"].is_unique,
        f"{split}: Patch_ID duplicado.",
    )

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    require(
        set(patches["Split"].unique()) == {split},
        (
            f"{split}: manifesto contém "
            "patch pertencente a outro split."
        ),
    )

    # ---------------------------------------------------------
    # Painting_ID válido
    # ---------------------------------------------------------

    pinturas_validas = set(
        df_pinturas["Painting_ID"]
    )

    pinturas_dos_patches = set(
        patches["Painting_ID"]
    )

    require(
        pinturas_dos_patches
        <= pinturas_validas,
        (
            f"{split}: existem patches associados "
            "a pinturas inexistentes no manifesto original."
        ),
    )

    # Toda pintura deve gerar pelo menos um patch.
    require(
        pinturas_validas
        == pinturas_dos_patches,
        (
            f"{split}: existem pinturas sem patches "
            "ou pinturas inesperadas."
        ),
    )

    # ---------------------------------------------------------
    # Dimensão dos patches
    # ---------------------------------------------------------

    larguras = (
        patches["x1"]
        - patches["x0"]
    )

    alturas = (
        patches["y1"]
        - patches["y0"]
    )

    require(
        (larguras == PATCH_SIZE).all(),
        (
            f"{split}: existe patch "
            "com largura diferente de "
            f"{PATCH_SIZE}."
        ),
    )

    require(
        (alturas == PATCH_SIZE).all(),
        (
            f"{split}: existe patch "
            "com altura diferente de "
            f"{PATCH_SIZE}."
        ),
    )

    # ---------------------------------------------------------
    # Coordenadas não negativas
    # ---------------------------------------------------------

    require(
        (patches["x0"] >= 0).all(),
        f"{split}: x0 negativo.",
    )

    require(
        (patches["y0"] >= 0).all(),
        f"{split}: y0 negativo.",
    )

    # ---------------------------------------------------------
    # Dimensões originais das pinturas
    # ---------------------------------------------------------

    dimensoes = (
        df_pinturas[
            [
                "Painting_ID",
                "Largura",
                "Altura",
            ]
        ]
        .set_index("Painting_ID")
    )

    patches["_Largura"] = (
        patches["Painting_ID"]
        .map(dimensoes["Largura"])
    )

    patches["_Altura"] = (
        patches["Painting_ID"]
        .map(dimensoes["Altura"])
    )

    require(
        patches["_Largura"].notna().all(),
        (
            f"{split}: não foi possível recuperar "
            "a largura de alguma pintura."
        ),
    )

    require(
        patches["_Altura"].notna().all(),
        (
            f"{split}: não foi possível recuperar "
            "a altura de alguma pintura."
        ),
    )

    # ---------------------------------------------------------
    # Nenhum patch ultrapassa a pintura
    # ---------------------------------------------------------

    require(
        (
            patches["x1"]
            <= patches["_Largura"]
        ).all(),
        (
            f"{split}: existe patch "
            "ultrapassando a largura da pintura."
        ),
    )

    require(
        (
            patches["y1"]
            <= patches["_Altura"]
        ).all(),
        (
            f"{split}: existe patch "
            "ultrapassando a altura da pintura."
        ),
    )

    # ---------------------------------------------------------
    # Cobertura das bordas
    # ---------------------------------------------------------

    por_pintura = patches.groupby(
        "Painting_ID"
    )

    menor_x = por_pintura["x0"].min()

    menor_y = por_pintura["y0"].min()

    maior_x = por_pintura["x1"].max()

    maior_y = por_pintura["y1"].max()

    require(
        (menor_x == 0).all(),
        (
            f"{split}: alguma pintura "
            "não começa a cobertura em x=0."
        ),
    )

    require(
        (menor_y == 0).all(),
        (
            f"{split}: alguma pintura "
            "não começa a cobertura em y=0."
        ),
    )

    largura_original = (
        dimensoes
        .loc[maior_x.index, "Largura"]
    )

    altura_original = (
        dimensoes
        .loc[maior_y.index, "Altura"]
    )

    require(
        (
            maior_x.values
            == largura_original.values
        ).all(),
        (
            f"{split}: alguma pintura "
            "não possui cobertura até a borda direita."
        ),
    )

    require(
        (
            maior_y.values
            == altura_original.values
        ).all(),
        (
            f"{split}: alguma pintura "
            "não possui cobertura até a borda inferior."
        ),
    )

    print(
        f"{split}: manifesto validado com sucesso "
        f"({len(patches):,} patches)."
    )

def preparar_imagem_opencv(
    imagem_pil,
) -> np.ndarray:
    """
    Converte a imagem fornecida pelo WikiArt/Hugging Face
    para uma matriz RGB compatível com OpenCV.

    A representação permanece em RGB para facilitar o uso
    posterior com modelos pré-treinados.
    """

    imagem_rgb = np.array(
        imagem_pil.convert("RGB")
    )

    require(
        imagem_rgb.ndim == 3,
        "Imagem deve possuir três dimensões.",
    )

    require(
        imagem_rgb.shape[2] == 3,
        "Imagem deve possuir três canais RGB.",
    )

    return imagem_rgb


def extrair_patch_opencv(
    imagem: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    """
    Extrai uma região 256x256 usando as coordenadas
    registradas no manifesto.
    """

    altura, largura = imagem.shape[:2]

    require(
        0 <= x0 < x1 <= largura,
        (
            "Coordenadas X inválidas: "
            f"x0={x0}, x1={x1}, largura={largura}"
        ),
    )

    require(
        0 <= y0 < y1 <= altura,
        (
            "Coordenadas Y inválidas: "
            f"y0={y0}, y1={y1}, altura={altura}"
        ),
    )

    # A matriz é manipulada com OpenCV/Numpy.
    patch = imagem[
        y0:y1,
        x0:x1,
    ].copy()

    require(
        patch.shape[:2]
        == (PATCH_SIZE, PATCH_SIZE),
        (
            "Patch com dimensão incorreta: "
            f"{patch.shape[:2]}"
        ),
    )

    return patch

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata-dir",
        default="data/metadata",
    )

    args = parser.parse_args()

    metadata_dir = Path(
        args.metadata_dir
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumo_total = []

    total_geral = 0

    print(
        f"PATCH_SIZE = {PATCH_SIZE}"
    )

    print(
        f"STRIDE = {STRIDE}"
    )

    print(
        "Política de borda: "
        "reposicionamento sem padding"
    )

    for split, arquivo_split in SPLIT_FILES.items():

        caminho_entrada = (
            metadata_dir
            / arquivo_split
        )

        caminho_saida = (
            metadata_dir
            / f"patches_{split}.csv.gz"
        )

        print(
            f"\nGerando {split}..."
        )

        df = pd.read_csv(
            caminho_entrada
        )

        resumo, total = gerar_manifesto_split(
            df=df,
            split=split,
            caminho_saida=caminho_saida,
        )

        validar_manifesto_patches(
        caminho_manifesto=caminho_saida,
        df_pinturas=df,
        split=split,
        )

        resumo_total.extend(
            resumo
        )

        total_geral += total

        tamanho_mb = (
            caminho_saida.stat().st_size
            / (1024 ** 2)
        )

        print(
            f"Pinturas: {len(df):,}"
        )

        print(
            f"Patches: {total:,}"
        )

        print(
            f"Arquivo: {caminho_saida}"
        )

        print(
            f"Tamanho comprimido: "
            f"{tamanho_mb:.2f} MB"
        )

    df_resumo = pd.DataFrame(
        resumo_total,
        columns=SUMMARY_COLUMNS,
    )

    caminho_resumo = (
        metadata_dir
        / "patches_resumo.csv"
    )

    df_resumo.to_csv(
        caminho_resumo,
        index=False,
        encoding="utf-8",
    )

    require(
        len(df_resumo) == 10_000,
        (
            "Resumo deve conter "
            "exatamente 10.000 pinturas."
        ),
    )

    esperado_total = sum(
        EXPECTED_PATCHES.values()
    )

    require(
        total_geral == esperado_total,
        (
            "Total geral inesperado: "
            f"{total_geral:,} vs "
            f"{esperado_total:,}"
        ),
    )

    print(
        "\nResumo por split:"
    )

    print(
        df_resumo
        .groupby("Split")["N_Patches"]
        .agg(
            [
                "count",
                "sum",
                "mean",
                "min",
                "max",
            ]
        )
        .round(2)
    )

    print(
        "\nPatches por macroclasse:"
    )

    print(
        df_resumo
        .groupby(
            [
                "Split",
                "Macro_Estilo",
            ]
        )["N_Patches"]
        .sum()
        .unstack()
    )

    print(
        f"\nTotal geral: "
        f"{total_geral:,} patches"
    )

    print(
        "\nGrade e manifestos "
        "gerados com sucesso."
    )


if __name__ == "__main__":
    main()