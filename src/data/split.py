"""Etapa 4 - Divisão determinística das pinturas em treino, validação e teste.

Regras:
- A divisão é feita no nível da PINTURA (antes de gerar patches).
- Estratificada por macroclasse E por estilo original (preserva a composição
  dos subestilos de Renaissance e Cubism em cada conjunto).
- Determinística: não usa sklearn nem pandas.sample(). Cada pintura recebe uma
  chave SHA-256 calculada a partir de SPLIT_SEED + Painting_ID, e a ordenação
  por essa chave define quem vai para cada conjunto.
- Contagens por macroclasse exatas (método dos maiores restos entre subestilos).

Uso:
    python -m src.data.split --manifest data/metadata/dataset_final.csv \
        --out data/metadata
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO - ajuste os nomes das colunas ao seu dataset_final.csv
# ---------------------------------------------------------------------------
SPLIT_SEED = 42
FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}

COL_ID = "Painting_ID"
COL_INDEX = "Indice_Dataset"
COL_CLASS = "Macro_Estilo"
COL_STYLE = "Estilo"
COL_GENRE = "Genero"
COL_ARTIST = "Artista"
COL_SHA = "Hash_SHA256_Pixels"
# ---------------------------------------------------------------------------


def split_key(painting_id: str, seed: int = SPLIT_SEED) -> str:
    """Chave determinística de ordenação (não depende de RNG)."""
    return hashlib.sha256(f"split|{seed}|{painting_id}".encode()).hexdigest()


def largest_remainder(weights: dict, total: int) -> dict:
    """Distribui `total` unidades proporcionalmente a `weights` (inteiros exatos)."""
    s = sum(weights.values())
    exact = {k: total * v / s for k, v in weights.items()}
    base = {k: int(x) for k, x in exact.items()}
    rest = total - sum(base.values())
    order = sorted(exact, key=lambda k: (exact[k] - base[k], k), reverse=True)
    for k in order[:rest]:
        base[k] += 1
    return base


def split_counts(n: int, fractions: dict) -> dict:
    """Quantidade de pinturas por conjunto para uma macroclasse de tamanho n."""
    return largest_remainder(fractions, n)


def make_split(df: pd.DataFrame, fractions: dict = FRACTIONS, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Retorna df com a coluna `Split` ('train' | 'validation' | 'test')."""
    df = df.copy()
    df["_key"] = df[COL_ID].map(lambda x: split_key(x, seed))
    df["Split"] = None
    names = list(fractions)

    for cls, g_cls in df.groupby(COL_CLASS, sort=True):
        per_split = split_counts(len(g_cls), fractions)  # ex.: {'train':1750,...}
        sub_sizes = g_cls.groupby(COL_STYLE).size().to_dict()

        # quantas pinturas de cada subestilo vão para cada conjunto
        alloc = {s: {} for s in sub_sizes}
        remaining = dict(sub_sizes)
        for name in names[:-1]:
            a = largest_remainder(remaining, per_split[name])
            for sub, k in a.items():
                alloc[sub][name] = k
                remaining[sub] -= k
        for sub, k in remaining.items():       # último conjunto recebe o resto
            alloc[sub][names[-1]] = k

        for sub, g_sub in g_cls.groupby(COL_STYLE):
            ordered = g_sub.sort_values("_key")
            start = 0
            for name in names:
                k = alloc[sub][name]
                df.loc[ordered.index[start:start + k], "Split"] = name
                start += k

    return df.drop(columns="_key")


def validate(
    df: pd.DataFrame,
    fractions: dict = FRACTIONS,
    seed: int = SPLIT_SEED,
) -> dict:
    """
    Executa verificações automáticas de integridade,
    reprodutibilidade e data leakage.
    """

    colunas_obrigatorias = {
        COL_ID,
        COL_INDEX,
        COL_CLASS,
        COL_STYLE,
        COL_GENRE,
        COL_SHA,
        "Split",
    }

    faltantes = colunas_obrigatorias - set(df.columns)

    if faltantes:
        raise AssertionError(
            f"Colunas obrigatórias ausentes: {sorted(faltantes)}"
        )

    if not df["Split"].notna().all():
        raise AssertionError(
            "Há pinturas sem conjunto."
        )

    splits_invalidos = (
        set(df["Split"].unique())
        - set(fractions.keys())
    )

    if splits_invalidos:
        raise AssertionError(
            f"Splits inesperados: {sorted(splits_invalidos)}"
        )

    # Verificação por três identificadores independentes.
    identificadores = [
        COL_ID,
        COL_INDEX,
        COL_SHA,
    ]

    nomes_splits = list(fractions.keys())

    for coluna in identificadores:

        if not df[coluna].is_unique:
            raise AssertionError(
                f"{coluna} repetido no conjunto completo."
            )

        partes = {
            split: set(
                df.loc[
                    df["Split"] == split,
                    coluna
                ]
            )
            for split in nomes_splits
        }

        for i, split_a in enumerate(nomes_splits):
            for split_b in nomes_splits[i + 1:]:

                sobreposicao = (
                    partes[split_a]
                    & partes[split_b]
                )

                if sobreposicao:
                    raise AssertionError(
                        f"Data leakage por {coluna}: "
                        f"{split_a} e {split_b} compartilham "
                        f"{len(sobreposicao)} registro(s)."
                    )

        total_splits = sum(
            len(parte)
            for parte in partes.values()
        )

        if total_splits != len(df):
            raise AssertionError(
                f"Soma dos splits por {coluna} "
                "difere do total."
            )

    # Verifica as quantidades de cada macroclasse.
    for macroclasse, grupo in df.groupby(COL_CLASS):

        esperado = split_counts(
            len(grupo),
            fractions,
        )

        real = (
            grupo["Split"]
            .value_counts()
            .to_dict()
        )

        correto = all(
            real.get(split, 0) == esperado[split]
            for split in fractions
        )

        if not correto:
            raise AssertionError(
                f"Contagem por split incorreta em "
                f"{macroclasse}: {real} vs {esperado}"
            )

    # Teste de reprodutibilidade.
    novamente = make_split(
        df.drop(columns="Split"),
        fractions=fractions,
        seed=seed,
    )

    split_atual = (
        df
        .set_index(COL_ID)["Split"]
        .sort_index()
    )

    split_recalculado = (
        novamente
        .set_index(COL_ID)["Split"]
        .sort_index()
    )

    if not split_atual.equals(split_recalculado):
        raise AssertionError(
            "A divisão não é determinística."
        )

    return {"ok": True}


def genre_distribution_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compara a distribuição de gênero de cada split
    com a distribuição da respectiva macroclasse
    no conjunto completo.

    TVD_pp:
        distância de variação total em pontos percentuais.

    max_diff_pp:
        maior diferença absoluta observada em um
        único gênero, em pontos percentuais.
    """

    resultados = []

    for macroclasse, grupo_classe in df.groupby(
        COL_CLASS,
        sort=True,
    ):

        distribuicao_base = (
            grupo_classe[COL_GENRE]
            .value_counts(normalize=True)
            .mul(100)
        )

        for split in FRACTIONS:

            grupo_split = grupo_classe[
                grupo_classe["Split"] == split
            ]

            distribuicao_split = (
                grupo_split[COL_GENRE]
                .value_counts(normalize=True)
                .mul(100)
            )

            generos = (
                distribuicao_base.index
                .union(distribuicao_split.index)
            )

            base = distribuicao_base.reindex(
                generos,
                fill_value=0.0,
            )

            atual = distribuicao_split.reindex(
                generos,
                fill_value=0.0,
            )

            diferencas = (
                atual - base
            ).abs()

            resultados.append({
                COL_CLASS: macroclasse,
                "Split": split,
                "N": len(grupo_split),
                "TVD_pp": diferencas.sum() / 2,
                "max_diff_pp": diferencas.max(),
                "Genero_maior_diferenca": diferencas.idxmax(),
            })

    return (
        pd.DataFrame(resultados)
        .round({
            "TVD_pp": 2,
            "max_diff_pp": 2,
        })
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/metadata/dataset_final.csv")
    ap.add_argument("--out", default="data/metadata")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    out = make_split(
    df,
    fractions=FRACTIONS,
    seed=SPLIT_SEED,
    )
    validate(
    out,
    fractions=FRACTIONS,
    seed=SPLIT_SEED,
    )

    out_dir = Path(args.out)
    for name in FRACTIONS:
        out[out.Split == name].drop(columns="Split").to_csv(out_dir / f"{name}.csv", index=False)
    print(out.groupby([COL_CLASS, "Split"]).size().unstack())
    print("\nPreservação da distribuição de gênero:")

    print(
        genre_distribution_report(out)
        .to_string(index=False)
    )

    print("\nValidações concluídas sem data leakage.")


if __name__ == "__main__":
    main()
