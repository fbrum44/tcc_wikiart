"""Testes da Etapa 11 que não exigem torch: reconstrução, métricas, descritores e seleção."""
import numpy as np
import pandas as pd
import pytest

from src.data.labels import CLASSES_FINAIS
from src.interpretability.mapas import (
    descritores_visuais,
    metricas_distribuicao,
    normalizar_mapa,
    reconstruir_mapa,
    sobrepor,
)
from src.interpretability.selecao import chave_hash, selecionar_exemplos


def grade(altura, largura, stride=128, patch=256):
    xs = list(range(0, largura - patch + 1, stride))
    ys = list(range(0, altura - patch + 1, stride))
    if xs[-1] + patch < largura:
        xs.append(largura - patch)
    if ys[-1] + patch < altura:
        ys.append(altura - patch)
    return [(x, y) for y in ys for x in xs]


def test_reconstrucao_de_mapa_constante():
    h, w = 500, 700
    g = grade(h, w)
    cams = [np.full((8, 8), 3.0, np.float32) for _ in g]
    mapa = reconstruir_mapa([x for x, _ in g], [y for _, y in g], cams, h, w)
    assert mapa.shape == (h, w)
    assert np.allclose(mapa, 3.0)


def test_reconstrucao_posiciona_ativacao_no_lugar_certo():
    h, w = 512, 512
    g = grade(h, w)
    cams = []
    for x, y in g:
        c = np.zeros((8, 8), np.float32)
        if (x, y) == (256, 256):
            c[:] = 1.0
        cams.append(c)
    mapa = reconstruir_mapa([x for x, _ in g], [y for _, y in g], cams, h, w)
    assert mapa[400, 400] > 0 and mapa[10, 10] == 0


def test_reconstrucao_falha_se_nao_cobre():
    with pytest.raises(ValueError):
        reconstruir_mapa([0], [0], [np.ones((8, 8), np.float32)], 600, 600)


def test_normalizacao_e_mapa_zero():
    m = np.random.default_rng(0).random((50, 60)).astype(np.float32) * 7
    n = normalizar_mapa(m)
    assert n.min() >= 0 and n.max() <= 1
    assert normalizar_mapa(np.zeros((5, 5), np.float32)).max() == 0


def test_metricas_borda_e_concentracao():
    m = np.zeros((100, 100), np.float32)
    m[0:5, :] = 1.0  # tudo na borda
    r = metricas_distribuicao(m)
    assert r["massa_borda"] == pytest.approx(1.0)
    m2 = np.zeros((100, 100), np.float32)
    m2[45:55, 45:55] = 1.0  # tudo no centro
    r2 = metricas_distribuicao(m2)
    assert r2["massa_borda"] == pytest.approx(0.0)
    assert r2["massa_top"] == pytest.approx(1.0)
    assert r2["cx"] == pytest.approx(0.5, abs=0.02)
    assert np.isnan(metricas_distribuicao(np.zeros((10, 10)))["massa_top"])


def test_descritores_regiao_de_alto_contraste():
    rng = np.random.default_rng(1)
    img = np.full((200, 200, 3), 128, np.uint8)
    img[:, :100] = rng.integers(0, 255, (200, 100, 3), dtype=np.uint8)  # metade esquerda ruidosa
    mapa = np.zeros((200, 200), np.float32)
    mapa[:, :100] = 1.0
    d = descritores_visuais(img, mapa, quantil=0.5)
    assert d["contraste_razao"] > 2
    assert d["bordas_razao"] > 2
    assert np.isnan(descritores_visuais(img, np.zeros((200, 200), np.float32))["contraste_razao"])


def test_sobrepor_formato():
    img = np.zeros((20, 30, 3), np.uint8)
    out = sobrepor(img, np.ones((20, 30), np.float32))
    assert out.shape == img.shape and out.dtype == np.uint8


def _dados_selecao():
    rng = np.random.default_rng(0)
    linhas, meta = [], []
    generos = ["religious_painting", "portrait", "landscape", "still_life"]
    for i in range(400):
        c = i % 4
        erro = (i % 7 == 0)
        p = (c + 1) % 4 if erro else c
        pid = f"wikiart_{i:04d}"
        linhas.append({"Painting_ID": pid, "true_idx": c, "true_class": CLASSES_FINAIS[c],
                       "pred_mean_idx": p, "pred_mean_class": CLASSES_FINAIS[p]}
                      | {f"mean_prob_{k}": 0.25 for k in CLASSES_FINAIS})
        meta.append({"Painting_ID": pid, "Indice_Dataset": i, "Genero": generos[int(rng.integers(0, 4))]})
    return pd.DataFrame(linhas), pd.DataFrame(meta)


def test_selecao_deterministica_e_sem_duplicatas():
    pred, meta = _dados_selecao()
    a = selecionar_exemplos(pred, meta, seed=42)
    b = selecionar_exemplos(pred.sample(frac=1, random_state=3), meta.sample(frac=1, random_state=5), seed=42)
    assert list(a.Painting_ID) == list(b.Painting_ID)
    assert a.Painting_ID.is_unique
    assert set(a.true_class) == set(CLASSES_FINAIS)
    assert a.acertou.any() and (~a.acertou).any()
    outra = selecionar_exemplos(pred, meta, seed=7)
    assert list(outra.Painting_ID) != list(a.Painting_ID)


def test_selecao_nao_depende_de_probabilidade():
    pred, meta = _dados_selecao()
    a = selecionar_exemplos(pred, meta, seed=42)
    pred2 = pred.copy()
    pred2["mean_prob_Baroque"] = 0.9
    b = selecionar_exemplos(pred2, meta, seed=42)
    assert list(a.Painting_ID) == list(b.Painting_ID)


def test_chave_hash_estavel():
    assert chave_hash("x", 42) == chave_hash("x", 42)
    assert chave_hash("x", 42) != chave_hash("x", 43)
