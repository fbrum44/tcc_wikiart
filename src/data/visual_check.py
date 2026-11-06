"""Etapa 6 - Validação visual das amostras depois das transformações.

Gera imagens PNG para conferir a olho:
  * patches de treino sem e com augmentation (a mesma posição, lado a lado);
  * patches de validação/teste (devem ser idênticos ao recorte original);
  * ida e volta da normalização (desnormalizar deve devolver o patch).

Uso (na raiz do projeto):
    python -m src.data.visual_check

Por padrão usa a augmentation da Etapa 5 (src/data/augmentations.py, via
src.data.patch_dataset:augment_treino). Para usar outra função, passe
--augment modulo:funcao; ela deve receber um patch uint8 RGB (256, 256, 3) e devolver
o patch transformado.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import cv2
import numpy as np

from .patch_dataset import PatchDataset, WikiArtImageSource, load_patch_manifest
from .normalization import IMAGENET_MEAN, IMAGENET_STD, denormalize_patch


def make_grid(images: list[np.ndarray], ncols: int, pad: int = 4) -> np.ndarray:
    h, w = images[0].shape[:2]
    nrows = -(-len(images) // ncols)
    grid = np.full((nrows * (h + pad) + pad, ncols * (w + pad) + pad, 3), 255, np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, ncols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        grid[y:y + h, x:x + w] = im
    return grid


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Não consegui gravar {path}")


def save_augmentation_grid(manifest, image_source, augment, out_path, n=8, seed=0,
                           mean=IMAGENET_MEAN, std=IMAGENET_STD) -> Path:
    """Linha de cima: patches originais. Linha de baixo: os mesmos patches após a augmentation
    e após ida e volta pela normalização (o que o modelo realmente vê)."""
    ds = PatchDataset(manifest, image_source, is_train=True, augment=augment,
                      mean=mean, std=std, to_tensor=False)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    originais = [ds.get_raw_patch(int(i)) for i in idx]
    vistos = [denormalize_patch(ds[int(i)]["image"], mean, std) for i in idx]
    out = Path(out_path)
    _save_rgb(out, make_grid(originais + vistos, ncols=len(idx)))
    return out


def save_eval_grid(manifest, image_source, out_path, n=8, seed=0,
                   mean=IMAGENET_MEAN, std=IMAGENET_STD) -> tuple[Path, int]:
    """Validação/teste: o patch visto pelo modelo deve ser igual ao recorte original.
    Devolve o caminho e a maior diferença de pixel (esperado: no máximo 1 nível)."""
    ds = PatchDataset(manifest, image_source, is_train=False, mean=mean, std=std, to_tensor=False)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    originais = [ds.get_raw_patch(int(i)) for i in idx]
    vistos = [denormalize_patch(ds[int(i)]["image"], mean, std) for i in idx]
    maior = max(int(np.abs(a.astype(int) - b.astype(int)).max()) for a, b in zip(originais, vistos))
    out = Path(out_path)
    _save_rgb(out, make_grid(originais + vistos, ncols=len(idx)))
    return out, maior


def _resolve(spec: str):
    modulo, nome = spec.split(":")
    return getattr(importlib.import_module(modulo), nome)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default="data/metadata")
    ap.add_argument("--out-dir", default="outputs/visual_check")
    ap.add_argument("--augment", default="src.data.patch_dataset:augment_treino",
                    help="modulo:funcao que recebe e devolve um patch (padrão: augmentation da Etapa 5)")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    src = WikiArtImageSource()
    augment = _resolve(args.augment) if args.augment else None
    base, out = Path(args.manifest_dir), Path(args.out_dir)

    train = load_patch_manifest(base / "patches_train.csv.gz")
    p = save_augmentation_grid(train, src, augment, out / "treino_original_vs_modelo.png", n=args.n)
    print("treino:", p)
    for nome in ("validation", "test"):
        df = load_patch_manifest(base / f"patches_{nome}.csv.gz")
        p, maior = save_eval_grid(df, src, out / f"{nome}_original_vs_modelo.png", n=args.n)
        print(f"{nome}: {p} | maior diferença de pixel: {maior}")


if __name__ == "__main__":
    main()
