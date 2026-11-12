"""Etapa 11 - Grad-CAM na ResNet-50 sobre pinturas do conjunto de teste.

Fluxo:
  predições por pintura (Etapa 9) -> seleção reproduzível -> para cada pintura selecionada:
  recorta TODOS os patches (grade da Etapa 5) -> Grad-CAM por patch (classe-alvo) ->
  reconstrói o mapa da pintura -> figuras, métricas e descritores.

Nenhum treinamento e nenhum ajuste: os mapas são calculados com o checkpoint final.
Classe-alvo: a classe prevista para a pintura (média das probabilidades). Quando a pintura é
errada, também é gerado o mapa da classe real, para comparar "onde o modelo olhou para a
classe que previu" e "para a classe correta".

Exemplo:
  python -m src.interpretability.run_gradcam --checkpoint models/resnet50_finetune_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.labels import CLASSES_FINAIS
from src.data.normalization import normalize_patch
from src.data.patch_dataset import WikiArtImageSource, load_patch_manifest
from src.evaluation.evaluate_model import PROB_COLUMNS, carregar_modelo
from src.interpretability.gradcam import GradCAM, camada_alvo_resnet
from src.interpretability.mapas import (
    descritores_visuais,
    metricas_distribuicao,
    normalizar_mapa,
    reconstruir_mapa,
    sobrepor,
)
from src.interpretability.selecao import selecionar_exemplos

PATCH = 256


def mapas_por_patch(modelo, gradcam, img, x0s, y0s, mean, std, alvos, device, batch_size):
    """Grad-CAM de todos os patches de uma pintura para cada alvo em `alvos`.

    Retorna ({alvo: cams (n,h,w)}, probs (n,4)).
    """
    n = len(x0s)
    cams = {a: [] for a in alvos}
    probs = []
    for ini in range(0, n, batch_size):
        fim = min(ini + batch_size, n)
        lote = np.stack([
            normalize_patch(np.ascontiguousarray(img[y0:y0 + PATCH, x0:x0 + PATCH]), mean, std)
            for x0, y0 in zip(x0s[ini:fim], y0s[ini:fim])
        ])
        x = torch.from_numpy(lote).to(device)
        for k, alvo in enumerate(alvos):
            c, logits = gradcam(x, int(alvo))
            cams[alvo].append(c)
            if k == 0:
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return {a: np.concatenate(v) for a, v in cams.items()}, np.concatenate(probs)


def figura_pintura(img, mapa_pred, mapa_true, linha, caminho, dpi):
    paineis = [("Pintura", img), (f"Grad-CAM: classe prevista ({linha.pred_mean_class})", sobrepor(img, mapa_pred))]
    if mapa_true is not None:
        paineis.append((f"Grad-CAM: classe real ({linha.true_class})", sobrepor(img, mapa_true)))
    fig, eixos = plt.subplots(1, len(paineis), figsize=(5.2 * len(paineis), 5.6))
    for ax, (titulo, im) in zip(np.atleast_1d(eixos), paineis):
        ax.imshow(im)
        ax.set_title(titulo, fontsize=10)
        ax.axis("off")
    status = "ACERTO" if linha.acertou else "ERRO"
    prob = linha[f"mean_prob_{linha.pred_mean_class}"]
    fig.suptitle(
        f"{status} | real: {linha.true_class} | previsto: {linha.pred_mean_class} ({prob:.0%}) | gênero: {linha.Genero}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figura_patches(patches, cams, cams_peak_idx, linha, caminho, dpi):
    k = len(cams_peak_idx)
    fig, eixos = plt.subplots(2, k, figsize=(2.6 * k, 5.4))
    eixos = np.asarray(eixos).reshape(2, k)
    for j, i in enumerate(cams_peak_idx):
        cam = cv2.resize(cams[i], (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)
        topo = cam.max()
        cam01 = cam / topo if topo > 1e-12 else cam
        eixos[0, j].imshow(patches[i])
        eixos[1, j].imshow(sobrepor(patches[i], cam01))
        eixos[0, j].set_title(f"patch {j + 1}", fontsize=9)
        for r in (0, 1):
            eixos[r, j].axis("off")
    fig.suptitle(f"{linha.Painting_ID} — patches de maior ativação (alvo: {linha.pred_mean_class})", fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def miniatura(im, lado=320):
    h, w = im.shape[:2]
    escala = lado / max(h, w)
    return cv2.resize(im, (max(1, int(w * escala)), max(1, int(h * escala))), interpolation=cv2.INTER_AREA)


def montagem(itens, titulo, caminho, dpi, colunas=4):
    if not itens:
        return
    linhas = int(np.ceil(len(itens) / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(3.4 * colunas, 3.6 * linhas))
    eixos = np.atleast_1d(eixos).ravel()
    for ax in eixos:
        ax.axis("off")
    for ax, (im, rotulo) in zip(eixos, itens):
        ax.imshow(im)
        ax.set_title(rotulo, fontsize=7)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/resnet50_finetune_best.pt")
    p.add_argument("--predictions", default="outputs/evaluation/resnet50/test_painting_predictions.csv.gz")
    p.add_argument("--patch-predictions", default="outputs/evaluation/resnet50/test_patch_predictions.csv.gz")
    p.add_argument("--out", default="outputs/gradcam/resnet50")
    p.add_argument("--n-por-grupo", type=int, default=4)
    p.add_argument("--n-por-genero", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pinturas", type=int, default=None, help="apenas para smoke test")
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    out = Path(args.out)
    for sub in ("paintings", "patches", "cams", "montagens"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    modelo, ckpt, mean, std = carregar_modelo("resnet50", Path(args.checkpoint), device)
    print("Checkpoint val_loss:", ckpt.get("val_loss"))
    gradcam = GradCAM(modelo, camada_alvo_resnet(modelo))

    pred = pd.read_csv(args.predictions)
    meta = pd.read_csv("data/metadata/test.csv")
    sel = selecionar_exemplos(
        pred, meta, n_por_grupo=args.n_por_grupo, n_por_genero=args.n_por_genero, seed=args.seed
    )
    if args.max_pinturas:
        sel = sel.head(args.max_pinturas)
    sel.drop(columns=["chave"]).to_csv(out / "selecao.csv", index=False)
    print(f"Pinturas selecionadas: {len(sel)} ({int(sel.acertou.sum())} acertos, {int((~sel.acertou).sum())} erros)")

    manifesto = load_patch_manifest("data/metadata/patches_test.csv.gz")
    manifesto = manifesto[manifesto["Painting_ID"].isin(set(sel["Painting_ID"]))]
    grupos = {pid: g for pid, g in manifesto.groupby("Painting_ID")}

    pred_patch_ref = None
    if Path(args.patch_predictions).exists():
        ref = pd.read_csv(args.patch_predictions, usecols=["Patch_ID", "pred_idx"])
        pred_patch_ref = dict(zip(ref["Patch_ID"], ref["pred_idx"]))
        print("Predições por patch da Etapa 9 carregadas para checagem de consistência.")
    else:
        print("AVISO: predições por patch não encontradas; checagem de consistência por patch ignorada.")

    fonte = WikiArtImageSource()
    registros, montagens = [], {}
    concordancia_patch = []
    reproduz_pintura = []

    for n, linha in enumerate(sel.itertuples(index=False), start=1):
        linha = pd.Series(linha._asdict())
        g = grupos[linha.Painting_ID].reset_index(drop=True)
        img = fonte(int(g["Indice_Dataset"].iloc[0]))
        h, w = img.shape[:2]
        x0s, y0s = g["x0"].to_numpy(), g["y0"].to_numpy()
        alvos = [int(linha.pred_mean_idx)]
        if int(linha.true_idx) != int(linha.pred_mean_idx):
            alvos.append(int(linha.true_idx))

        cams, probs = mapas_por_patch(modelo, gradcam, img, x0s, y0s, mean, std, alvos, device, args.batch_size)

        # consistência com a Etapa 9
        pred_idx = probs.argmax(axis=1)
        if pred_patch_ref is not None:
            ref = np.array([pred_patch_ref[pid] for pid in g["Patch_ID"]])
            concordancia_patch.append(float((ref == pred_idx).mean()))
        reproduz_pintura.append(int(probs.mean(axis=0).argmax()) == int(linha.pred_mean_idx))

        mapa_bruto = reconstruir_mapa(x0s, y0s, cams[alvos[0]], h, w)
        mapa01 = normalizar_mapa(mapa_bruto)
        mapa_true01 = None
        if len(alvos) > 1:
            mapa_true01 = normalizar_mapa(reconstruir_mapa(x0s, y0s, cams[alvos[1]], h, w))

        np.savez_compressed(
            out / "cams" / f"{linha.Painting_ID}.npz",
            patch_ids=g["Patch_ID"].to_numpy(dtype=str),
            x0=x0s, y0=y0s, altura=h, largura=w,
            alvo_pred=alvos[0], cams_pred=cams[alvos[0]],
            alvo_true=alvos[1] if len(alvos) > 1 else -1,
            cams_true=cams[alvos[1]] if len(alvos) > 1 else np.zeros((0,), np.float32),
            probs=probs,
        )

        figura_pintura(img, mapa01, mapa_true01, linha, out / "paintings" / f"{linha.Painting_ID}.png", args.dpi)

        pico = cams[alvos[0]].reshape(len(g), -1).max(axis=1)
        top = np.argsort(-pico)[:4]
        patches = [np.ascontiguousarray(img[y:y + PATCH, x:x + PATCH]) for x, y in zip(x0s, y0s)]
        figura_patches(patches, cams[alvos[0]], top, linha, out / "patches" / f"{linha.Painting_ID}.png", args.dpi)

        reg = {
            "Painting_ID": linha.Painting_ID, "true_class": linha.true_class,
            "pred_mean_class": linha.pred_mean_class, "acertou": bool(linha.acertou),
            "Genero": linha.Genero, "n_patches": len(g), "altura": h, "largura": w,
            "prob_classe_prevista": float(linha[f"mean_prob_{linha.pred_mean_class}"]),
            "motivos": linha.motivos, "ativacao_max_bruta": float(mapa_bruto.max()),
        }
        reg.update(metricas_distribuicao(mapa_bruto))
        reg.update(descritores_visuais(img, mapa01))
        registros.append(reg)

        img_s = miniatura(img)
        mapa_s = cv2.resize(mapa01, (img_s.shape[1], img_s.shape[0]), interpolation=cv2.INTER_AREA)
        miniat = sobrepor(img_s, mapa_s)
        chave = f"{linha.true_class}|{'acerto' if linha.acertou else 'erro'}"
        rotulo = f"prev: {linha.pred_mean_class[:6]} ({reg['prob_classe_prevista']:.0%}) {linha.Genero[:14]}"
        montagens.setdefault(chave, []).append((miniat, rotulo))
        print(f"[{n}/{len(sel)}] {linha.Painting_ID} {chave} patches={len(g)}")

    gradcam.remover()

    for chave, itens in montagens.items():
        classe, tipo = chave.split("|")
        montagem(itens, f"Grad-CAM (ResNet-50) — real: {classe} — {tipo}", out / "montagens" / f"{classe}_{tipo}.png", args.dpi)

    df = pd.DataFrame(registros)
    df.to_csv(out / "metricas_por_pintura.csv", index=False)

    colunas = ["massa_borda", "massa_top", "cx", "cy", "contraste_razao", "saturacao_razao", "brilho_razao", "bordas_razao"]
    resumo = df.groupby(["true_class", "acertou"])[colunas].agg(["mean", "median"]).round(3)
    resumo.columns = ["_".join(c) for c in resumo.columns]
    resumo = resumo.join(df.groupby(["true_class", "acertou"]).size().rename("n"))
    resumo.to_csv(out / "resumo_por_grupo.csv")

    figura_resumo(df, out / "resumo_distribuicao.png", args.dpi)

    verif = {
        "pinturas": len(df),
        "concordancia_patch_media": float(np.mean(concordancia_patch)) if concordancia_patch else None,
        "concordancia_patch_minima": float(np.min(concordancia_patch)) if concordancia_patch else None,
        "predicao_pintura_reproduzida": float(np.mean(reproduz_pintura)),
        "checkpoint": str(args.checkpoint),
        "camada_alvo": "layer4",
        "resolucao_mapa_por_patch": "8x8 (ampliado bilinearmente para 256x256)",
        "alvo": "classe prevista (média das probabilidades); classe real adicional nos erros",
        "normalizacao_mapa": "percentil 99,5 por pintura",
        "seed_selecao": args.seed,
    }
    (out / "verificacao.json").write_text(json.dumps(verif, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(verif, indent=2, ensure_ascii=False))
    print("Resultados em:", out)


def figura_resumo(df, caminho, dpi):
    metricas = [
        ("massa_borda", "Fração da ativação na faixa de 10% da borda"),
        ("massa_top", "Fração da ativação nos 10% de pixels mais ativos"),
        ("contraste_razao", "Contraste local: região ativada / resto"),
        ("bordas_razao", "Densidade de bordas: região ativada / resto"),
    ]
    fig, eixos = plt.subplots(1, len(metricas), figsize=(4.6 * len(metricas), 4.4))
    ordem = [c for c in CLASSES_FINAIS if c in set(df.true_class)]
    for ax, (col, titulo) in zip(eixos, metricas):
        dados, rotulos = [], []
        for c in ordem:
            for ac, nome in ((True, "acerto"), (False, "erro")):
                v = df[(df.true_class == c) & (df.acertou == ac)][col].dropna().to_numpy()
                if len(v):
                    dados.append(v)
                    rotulos.append(f"{c[:5]}\n{nome} (n={len(v)})")
        ax.boxplot(dados, showmeans=True)
        ax.set_xticks(range(1, len(rotulos) + 1))
        ax.set_xticklabels(rotulos, fontsize=6, rotation=60)
        ax.set_title(titulo, fontsize=8)
        if "razao" in col:
            ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
