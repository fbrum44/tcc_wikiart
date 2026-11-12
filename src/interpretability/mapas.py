"""Etapa 11 - Operações espaciais sobre mapas Grad-CAM (sem dependência de torch).

Os mapas de ativação são calculados por patch (8x8 para ResNet-50 com entrada 256x256).
Este módulo reconstrói o mapa da pintura inteira usando as coordenadas da Etapa 5
(mesma política da reconstrução de pixels: soma nas sobreposições dividida pelo número
de coberturas) e calcula medidas simples e reproduzíveis sobre o mapa.

Interpretação: o mapa indica regiões que aumentaram a pontuação da classe-alvo. Não prova
que o modelo "compreendeu" o conteúdo, nem estabelece causalidade.
"""
from __future__ import annotations

import cv2
import numpy as np

PATCH_SIZE = 256


def reconstruir_mapa(x0s, y0s, cams, altura: int, largura: int, patch: int = PATCH_SIZE) -> np.ndarray:
    """Une mapas por patch (n, h, w) em um mapa (altura, largura) da pintura.

    Cada mapa é ampliado para patch x patch (interpolação bilinear) e posicionado em (x0, y0).
    Nas áreas sobrepostas usa-se a média. Falha se algum pixel não for coberto.
    """
    soma = np.zeros((altura, largura), np.float32)
    cont = np.zeros((altura, largura), np.float32)
    for x0, y0, cam in zip(x0s, y0s, cams):
        x0, y0 = int(x0), int(y0)
        if x0 < 0 or y0 < 0 or x0 + patch > largura or y0 + patch > altura:
            raise ValueError(f"patch ({x0},{y0}) fora da pintura {largura}x{altura}")
        ampliado = cv2.resize(np.asarray(cam, np.float32), (patch, patch), interpolation=cv2.INTER_LINEAR)
        soma[y0:y0 + patch, x0:x0 + patch] += ampliado
        cont[y0:y0 + patch, x0:x0 + patch] += 1.0
    if cont.min() < 1:
        raise ValueError("a grade de patches não cobre toda a pintura")
    return soma / cont


def normalizar_mapa(mapa: np.ndarray, percentil: float = 99.5) -> np.ndarray:
    """Escala para [0, 1] usando um percentil alto (evita que um único pico domine)."""
    topo = float(np.percentile(mapa, percentil))
    if topo <= 1e-12:
        return np.zeros_like(mapa, np.float32)
    return np.clip(mapa / topo, 0.0, 1.0).astype(np.float32)


def metricas_distribuicao(mapa: np.ndarray, fracao_borda: float = 0.10, fracao_topo: float = 0.10) -> dict:
    """Medidas de como a ativação se distribui espacialmente (mapa bruto, não negativo).

    - massa_borda: fração da ativação total na faixa de `fracao_borda` em cada lado da pintura
      (a faixa cobre 1-(1-2f)^2 da área; compare com `area_borda`).
    - massa_top: fração da ativação total nos `fracao_topo` pixels mais ativos (concentração).
    - cx, cy: centro de massa normalizado (0 = esquerda/topo, 1 = direita/base).
    """
    m = np.clip(mapa.astype(np.float64), 0, None)
    total = m.sum()
    h, w = m.shape
    faixa = max(1, int(round(fracao_borda * min(h, w))))
    borda = np.ones_like(m, bool)
    borda[faixa:h - faixa, faixa:w - faixa] = False
    area_borda = float(borda.mean())
    if total <= 1e-12:
        return {"massa_borda": np.nan, "area_borda": area_borda, "massa_top": np.nan, "cx": np.nan, "cy": np.nan}
    k = max(1, int(round(fracao_topo * m.size)))
    topo = np.partition(m.ravel(), -k)[-k:].sum()
    ys, xs = np.mgrid[0:h, 0:w]
    return {
        "massa_borda": float(m[borda].sum() / total),
        "area_borda": area_borda,
        "massa_top": float(topo / total),
        "cx": float((m * xs).sum() / total / max(w - 1, 1)),
        "cy": float((m * ys).sum() / total / max(h - 1, 1)),
    }


def _contraste_local(cinza: np.ndarray, janela: int = 15) -> np.ndarray:
    c = cinza.astype(np.float32)
    media = cv2.blur(c, (janela, janela))
    media2 = cv2.blur(c * c, (janela, janela))
    return np.sqrt(np.clip(media2 - media * media, 0, None))


def descritores_visuais(imagem_rgb: np.ndarray, mapa01: np.ndarray, quantil: float = 0.80) -> dict:
    """Compara descritores simples entre a região de alta ativação e o resto da pintura.

    Região alta = pixels com mapa >= quantil `quantil` do próprio mapa (20% mais ativos).
    Descritores: contraste local (desvio-padrão em janela 15x15), saturação (HSV), brilho
    e densidade de bordas (Canny). Retorna alto, baixo e razão alto/baixo.
    Descritivo: uma razão > 1 indica que a região ativada tem mais daquele atributo; não
    indica que o modelo "use" o atributo.
    """
    if mapa01.shape != imagem_rgb.shape[:2]:
        raise ValueError("mapa e imagem com tamanhos diferentes")
    nomes = ("contraste", "saturacao", "brilho", "bordas")
    vazio = {f"{n}_{s}": np.nan for n in nomes for s in ("alto", "baixo", "razao")}
    if float(mapa01.max() - mapa01.min()) < 1e-9:
        return vazio
    limiar = np.quantile(mapa01, quantil)
    alta = mapa01 >= limiar
    baixa = ~alta
    if alta.sum() == 0 or baixa.sum() == 0:
        return vazio
    cinza = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2HSV)
    mapas = {
        "contraste": _contraste_local(cinza),
        "saturacao": hsv[..., 1].astype(np.float32) / 255.0,
        "brilho": cinza.astype(np.float32) / 255.0,
        "bordas": (cv2.Canny(cinza, 100, 200) > 0).astype(np.float32),
    }
    saida = {}
    for nome, valores in mapas.items():
        a, b = float(valores[alta].mean()), float(valores[baixa].mean())
        saida[f"{nome}_alto"] = a
        saida[f"{nome}_baixo"] = b
        saida[f"{nome}_razao"] = a / b if b > 1e-9 else np.nan
    return saida


def sobrepor(imagem_rgb: np.ndarray, mapa01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Sobrepõe o mapa (colormap 'jet') à imagem. Retorna uint8 RGB."""
    cor = cv2.applyColorMap((mapa01 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cor = cv2.cvtColor(cor, cv2.COLOR_BGR2RGB)
    mistura = (1 - alpha) * imagem_rgb.astype(np.float32) + alpha * cor.astype(np.float32)
    return np.clip(mistura, 0, 255).astype(np.uint8)
