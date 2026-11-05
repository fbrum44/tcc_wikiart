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


def validate(df: pd.DataFrame, fractions: dict = FRACTIONS) -> dict:
    """Testes automáticos contra vazamento (data leakage). Levanta AssertionError."""
    assert df["Split"].notna().all(), "Há pinturas sem conjunto"
    assert df[COL_ID].is_unique, "Painting_ID repetido"

    parts = {s: set(df.loc[df.Split == s, COL_ID]) for s in fractions}
    names = list(fractions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (parts[a] & parts[b]), f"Sobreposição de IDs entre {a} e {b}"
    assert sum(len(p) for p in parts.values()) == len(df), "Soma dos splits != total"

    if COL_SHA in df.columns:                  # mesmo conteúdo em splits diferentes?
        n_splits = df.groupby(COL_SHA)["Split"].nunique()
        assert (n_splits == 1).all(), "Mesmo SHA-256 em conjuntos diferentes"

    for cls, g in df.groupby(COL_CLASS):
        esperado = split_counts(len(g), fractions)
        real = g["Split"].value_counts().to_dict()
        assert all(real.get(s, 0) == esperado[s] for s in fractions), \
            f"Contagem por split incorreta em {cls}: {real} vs {esperado}"

    # reprodutibilidade: refazer a divisão deve dar o mesmo resultado
    again = make_split(df.drop(columns="Split"), fractions)
    assert (again.set_index(COL_ID)["Split"] == df.set_index(COL_ID)["Split"]).all(), \
        "Divisão não é determinística"
    return {"ok": True}


def distribution_report(df: pd.DataFrame, col: str, top: int = 8) -> pd.DataFrame:
    """% de cada categoria de `col` por conjunto (para comparar gênero/artista)."""
    tab = pd.crosstab(df[col], df["Split"], normalize="columns") * 100
    tab["max_diff_pp"] = tab.max(axis=1) - tab.min(axis=1)
    return tab.sort_values(tab.columns[0], ascending=False).head(top).round(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/metadata/dataset_final.csv")
    ap.add_argument("--out", default="data/metadata")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    out = make_split(df)
    validate(out)

    out_dir = Path(args.out)
    for name in FRACTIONS:
        out[out.Split == name].drop(columns="Split").to_csv(out_dir / f"{name}.csv", index=False)
    print(out.groupby([COL_CLASS, "Split"]).size().unstack())


if __name__ == "__main__":
    main()
