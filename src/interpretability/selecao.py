"""Etapa 11 - Seleção reproduzível das pinturas para análise com Grad-CAM.

A seleção NÃO usa a confiança do modelo nem a aparência do mapa: dentro de cada grupo as
pinturas são ordenadas por uma chave SHA-256 de (semente, Painting_ID), como na Etapa 3/4,
e as primeiras são escolhidas. Assim os exemplos não são escolhidos a dedo.

Grupos:
  1. cada macroclasse real x {acerto, erro} (predição por média das probabilidades);
  2. para Renaissance e Baroque, cada gênero de foco x {acerto, erro}.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from src.data.labels import CLASSES_FINAIS

CANDIDATAS_GENERO = ("Genero", "Gênero", "Genre", "genre")


def chave_hash(painting_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{painting_id}".encode("utf-8")).hexdigest()


def detectar_coluna_genero(meta: pd.DataFrame) -> str:
    for nome in CANDIDATAS_GENERO:
        if nome in meta.columns:
            return nome
    raise ValueError(f"Coluna de gênero não encontrada. Colunas disponíveis: {list(meta.columns)}")


def selecionar_exemplos(
    pred_pinturas: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    n_por_grupo: int = 4,
    generos_foco=("religious_painting", "portrait"),
    classes_foco=("Renaissance", "Baroque"),
    n_por_genero: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Devolve uma linha por pintura selecionada, com a coluna `motivos`."""
    col_genero = detectar_coluna_genero(meta)
    base = pred_pinturas.merge(
        meta[["Painting_ID", "Indice_Dataset", col_genero]].rename(columns={col_genero: "Genero"}),
        on="Painting_ID",
        how="left",
        validate="one_to_one",
    )
    if base["Genero"].isna().any():
        raise ValueError("Há pinturas sem gênero após o merge com o manifesto do teste.")
    base["acertou"] = base["pred_mean_idx"] == base["true_idx"]
    base["chave"] = base["Painting_ID"].map(lambda p: chave_hash(p, seed))

    escolhidas = []

    def pegar(filtro, motivo, n):
        sub = base[filtro].sort_values("chave").head(n).copy()
        sub["motivo"] = motivo
        escolhidas.append(sub)

    for classe in CLASSES_FINAIS:
        for acertou in (True, False):
            rot = "acerto" if acertou else "erro"
            pegar((base.true_class == classe) & (base.acertou == acertou), f"{classe}|{rot}", n_por_grupo)

    for classe in classes_foco:
        for genero in generos_foco:
            for acertou in (True, False):
                rot = "acerto" if acertou else "erro"
                pegar(
                    (base.true_class == classe) & (base.Genero == genero) & (base.acertou == acertou),
                    f"{classe}/{genero}|{rot}",
                    n_por_genero,
                )

    todas = pd.concat(escolhidas, ignore_index=True)
    final = (
        todas.groupby("Painting_ID", sort=False)
        .agg(motivos=("motivo", lambda s: ";".join(s)), **{c: (c, "first") for c in base.columns if c != "Painting_ID"})
        .reset_index()
    )
    return final.sort_values(["true_idx", "acertou", "chave"], ascending=[True, False, True]).reset_index(drop=True)
