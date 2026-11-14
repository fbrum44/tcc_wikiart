"""Etapa 11 - Comparação estatística entre acertos e erros a partir de metricas_por_pintura.csv.

Entrada: saída de run_gradcam (uma linha por pintura). Não recalcula nenhum mapa.

Medidas:
  massa_borda_rel = massa_borda / area_borda   (1 = ativação distribuída como a área da faixa;
                                                >1 = concentrada na borda da pintura)
  massa_top, contraste_razao, bordas_razao, saturacao_razao, brilho_razao (ver mapas.py)

Para cada classe real (e para todas juntas): mediana em acertos e em erros, diferença, teste de
Mann-Whitney (bilateral) e AUC = P(valor em erro > valor em acerto). p-valores corrigidos por
Holm sobre todos os testes da tabela. Com poucas pinturas por grupo os resultados são
exploratórios: não implicam causalidade nem que o modelo "use" o atributo.

Uso:
  python -m src.interpretability.analise_grupos --in outputs/gradcam/resnet50_amplo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

METRICAS = ["massa_borda_rel", "massa_top", "contraste_razao", "bordas_razao", "saturacao_razao", "brilho_razao"]
MIN_POR_GRUPO = 3


def holm(pvalores: np.ndarray) -> np.ndarray:
    """Correção de Holm-Bonferroni (NaN são ignorados)."""
    p = np.asarray(pvalores, float)
    saida = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    ordem = ok[np.argsort(p[ok])]
    m = len(ordem)
    corrigido = np.maximum.accumulate([(m - i) * p[j] for i, j in enumerate(ordem)])
    saida[ordem] = np.minimum(corrigido, 1.0)
    return saida


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["massa_borda_rel"] = df["massa_borda"] / df["area_borda"]
    df["acertou"] = df["acertou"].astype(bool)
    return df


def comparar_acerto_erro(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    estratos = [("todas", df)] + [(c, g) for c, g in df.groupby("true_class", sort=False)]
    for nome, g in estratos:
        for metrica in METRICAS:
            a = g.loc[g.acertou, metrica].dropna().to_numpy()
            e = g.loc[~g.acertou, metrica].dropna().to_numpy()
            reg = {
                "estrato": nome, "metrica": metrica, "n_acerto": len(a), "n_erro": len(e),
                "mediana_acerto": np.median(a) if len(a) else np.nan,
                "mediana_erro": np.median(e) if len(e) else np.nan,
                "p_mannwhitney": np.nan, "auc_erro_maior": np.nan,
            }
            reg["diferenca_medianas"] = reg["mediana_erro"] - reg["mediana_acerto"]
            if len(a) >= MIN_POR_GRUPO and len(e) >= MIN_POR_GRUPO:
                u, p = mannwhitneyu(e, a, alternative="two-sided")
                reg["p_mannwhitney"] = float(p)
                reg["auc_erro_maior"] = float(u / (len(e) * len(a)))
            linhas.append(reg)
    tab = pd.DataFrame(linhas)
    tab["p_holm"] = holm(tab["p_mannwhitney"].to_numpy())
    return tab


def comparar_entre_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis entre as classes, somente entre pinturas acertadas."""
    acertos = df[df.acertou]
    linhas = []
    for metrica in METRICAS:
        grupos = [g[metrica].dropna().to_numpy() for _, g in acertos.groupby("true_class")]
        grupos = [g for g in grupos if len(g) >= MIN_POR_GRUPO]
        reg = {"metrica": metrica, "n_classes": len(grupos), "p_kruskal": np.nan}
        for classe, g in acertos.groupby("true_class"):
            reg[f"mediana_{classe}"] = g[metrica].median()
        if len(grupos) >= 2:
            reg["p_kruskal"] = float(kruskal(*grupos).pvalue)
        linhas.append(reg)
    tab = pd.DataFrame(linhas)
    tab["p_holm"] = holm(tab["p_kruskal"].to_numpy())
    return tab


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="entrada", required=True, help="pasta de saída do run_gradcam")
    args = p.parse_args()

    pasta = Path(args.entrada)
    df = preparar(pd.read_csv(pasta / "metricas_por_pintura.csv"))
    print(f"Pinturas: {len(df)} ({int(df.acertou.sum())} acertos, {int((~df.acertou).sum())} erros)")
    print(df.groupby(["true_class", "acertou"]).size().unstack(fill_value=0))

    ae = comparar_acerto_erro(df)
    ae.round(4).to_csv(pasta / "comparacao_acerto_erro.csv", index=False)
    ec = comparar_entre_classes(df)
    ec.round(4).to_csv(pasta / "comparacao_entre_classes_acertos.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nAcerto x erro (Mann-Whitney; p_holm corrige todos os testes da tabela):")
    print(ae.round(3).to_string(index=False))
    print("\nEntre classes, só acertos (Kruskal-Wallis):")
    print(ec.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
