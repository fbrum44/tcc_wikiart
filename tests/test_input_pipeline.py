"""Testes da Etapa 6 (não precisam de PyTorch nem do WikiArt).

Rodar na raiz do projeto:
    python tests/test_input_pipeline.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.normalization import denormalize_patch, normalize_patch, IMAGENET_MEAN, IMAGENET_STD
from src.data.patch_dataset import (
    CLASS_NAMES, PaintingGroupedSampler, PatchDataset, build_loaders, load_patch_manifest,
)
from src.data import visual_check


# ---------------------------------------------------------------- dados sintéticos
def make_images():
    """8 pinturas de tamanhos diferentes (2 por macroclasse); indice -> imagem."""
    rng = np.random.default_rng(0)
    sizes = [(300, 300), (700, 400), (300, 300), (520, 900), (300, 300), (800, 800), (300, 300), (400, 640)]
    return {i: rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for i, (h, w) in enumerate(sizes)}


def grid_positions(n, stride=128, size=256):
    pos = list(range(0, n - size + 1, stride))
    if pos[-1] != n - size:
        pos.append(n - size)
    return pos


def make_manifest(images, split="train"):
    rows = []
    for idx, img in images.items():
        h, w = img.shape[:2]
        for y0 in grid_positions(h):
            for x0 in grid_positions(w):
                rows.append(dict(
                    Patch_ID=f"p{idx}_x{x0:05d}_y{y0:05d}", Painting_ID=f"p{idx}", Indice_Dataset=idx,
                    Split=split, Macro_Estilo=CLASS_NAMES[idx % 4], Estilo="s",
                    x0=x0, y0=y0, x1=x0 + 256, y1=y0 + 256))
    return pd.DataFrame(rows)


IMAGES = make_images()
SOURCE = lambda i: IMAGES[int(i)]


# ---------------------------------------------------------------- testes
def test_normalizacao_ida_e_volta():
    patch = np.random.default_rng(1).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    arr = normalize_patch(patch)
    assert arr.shape == (3, 256, 256) and arr.dtype == np.float32
    assert np.abs(denormalize_patch(arr).astype(int) - patch.astype(int)).max() <= 1


def test_normalizacao_valor_conhecido():
    cinza = np.empty((4, 4, 3), np.uint8)
    for c, m in enumerate(IMAGENET_MEAN):
        cinza[..., c] = round(m * 255)
    assert np.abs(normalize_patch(cinza)).max() < 0.02          # ~0 quando pixel == média
    branco = normalize_patch(np.full((4, 4, 3), 255, np.uint8))
    esperado = (1 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert abs(float(branco[0, 0, 0]) - esperado) < 1e-5
    try:
        normalize_patch(np.zeros((4, 4, 3), np.float32))
        raise AssertionError("deveria recusar float")
    except ValueError:
        pass


def test_recorte_igual_ao_manifesto():
    df = make_manifest(IMAGES)
    ds = PatchDataset(df, SOURCE, is_train=False, to_tensor=False)
    for i in np.random.default_rng(2).choice(len(ds), 50, replace=False):
        r = df.iloc[i]
        esperado = IMAGES[r.Indice_Dataset][r.y0:r.y1, r.x0:r.x1]
        assert np.array_equal(ds.get_raw_patch(int(i)), esperado)


def test_avaliacao_sem_transformacao():
    df = make_manifest(IMAGES, "validation")
    ds = PatchDataset(df, SOURCE, is_train=False, to_tensor=False)
    item = ds[7]
    assert np.abs(denormalize_patch(item["image"]).astype(int) - ds.get_raw_patch(7).astype(int)).max() <= 1
    assert item["label"] == CLASS_NAMES.index(df.iloc[7].Macro_Estilo)
    assert item["painting_id"] == df.iloc[7].Painting_ID and item["x0"] == df.iloc[7].x0


def test_augmentation_so_no_treino():
    df = make_manifest(IMAGES)
    flip = lambda p: np.ascontiguousarray(p[:, ::-1])
    try:
        PatchDataset(df, SOURCE, is_train=False, augment=flip)
        raise AssertionError("validação/teste não pode receber augmentation")
    except ValueError:
        pass
    ds = PatchDataset(df, SOURCE, is_train=True, augment=flip, to_tensor=False)
    visto = denormalize_patch(ds[3]["image"])
    assert np.abs(visto.astype(int) - flip(ds.get_raw_patch(3)).astype(int)).max() <= 1


def test_patch_fora_da_imagem():
    df = make_manifest(IMAGES)
    df.loc[0, ["x0", "x1"]] = [5000, 5256]
    ds = PatchDataset(df, SOURCE, is_train=False, to_tensor=False)
    try:
        ds[0]
        raise AssertionError("deveria falhar")
    except ValueError:
        pass


def test_amostrador_mesma_cota_por_pintura():
    df = make_manifest(IMAGES)
    contagem = df.groupby("Painting_ID").size()
    assert contagem.max() > 3 * contagem.min()                  # pinturas muito desiguais
    k = 3
    sampler = PaintingGroupedSampler(df["Painting_ID"], k, seed=42)
    idx = list(sampler)
    assert len(idx) == len(sampler) == k * df["Painting_ID"].nunique()
    assert len(set(idx)) == len(idx)                            # sem repetição na época
    por_pintura = df.iloc[idx].groupby("Painting_ID").size()
    assert (por_pintura == k).all()                             # pintura grande não pesa mais
    por_classe = df.iloc[idx].groupby("Macro_Estilo").size()
    assert por_classe.nunique() == 1                            # classes balanceadas
    pintura_seq = df.iloc[idx]["Painting_ID"].tolist()          # patches da mesma pintura juntos
    blocos = sum(a != b for a, b in zip(pintura_seq, pintura_seq[1:])) + 1
    assert blocos == df["Painting_ID"].nunique()


def test_amostrador_reprodutivel_e_muda_por_epoca():
    df = make_manifest(IMAGES)
    a, b = PaintingGroupedSampler(df["Painting_ID"], 4, 42), PaintingGroupedSampler(df["Painting_ID"], 4, 42)
    e1a, e1b = list(a), list(b)
    assert e1a == e1b                                           # mesma semente, mesma época
    assert list(a) != e1a                                       # época seguinte sorteia outros patches
    c = PaintingGroupedSampler(df["Painting_ID"], 4, 43)
    assert list(c) != e1a


def test_manifesto_gz():
    df = make_manifest(IMAGES)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "patches_train.csv.gz"
        df.to_csv(p, index=False)
        lido = load_patch_manifest(p)
        assert len(lido) == len(df)
        df.assign(Split=["a", "b"] * (len(df) // 2) + ["a"] * (len(df) % 2)).to_csv(p, index=False)
        try:
            load_patch_manifest(p)
            raise AssertionError("deveria recusar Split misto")
        except ValueError:
            pass


def test_validacao_visual_gera_png():
    df = make_manifest(IMAGES)
    flip = lambda p: np.ascontiguousarray(p[:, ::-1])
    with tempfile.TemporaryDirectory() as d:
        p = visual_check.save_augmentation_grid(df, SOURCE, flip, Path(d) / "a.png", n=4)
        assert p.exists() and p.stat().st_size > 0
        p2, maior = visual_check.save_eval_grid(df, SOURCE, Path(d) / "b.png", n=4)
        assert p2.exists() and maior <= 1


def test_augmentation_real_do_projeto():
    """Liga o dataset à augmentations.py da Etapa 5 (via augment_treino)."""
    from src.data.patch_dataset import augment_treino
    df = make_manifest(IMAGES)
    treino = PatchDataset(df, SOURCE, is_train=True, augment=augment_treino, to_tensor=False)
    alterados = 0
    for i in range(60):
        bruto = treino.get_raw_patch(i)
        visto = denormalize_patch(treino[i]["image"])
        assert visto.shape == bruto.shape and visto.dtype == np.uint8
        alterados += int(np.abs(visto.astype(int) - bruto.astype(int)).max() > 1)
    assert 10 < alterados < 60                                  # a augmentation age, mas nem sempre
    try:
        PatchDataset(df, SOURCE, is_train=False, augment=augment_treino)
        raise AssertionError("validação/teste não pode receber augmentation")
    except ValueError:
        pass
    val = PatchDataset(make_manifest(IMAGES, "validation"), SOURCE, is_train=False, to_tensor=False)
    assert np.abs(denormalize_patch(val[5]["image"]).astype(int) - val.get_raw_patch(5).astype(int)).max() <= 1

def test_manifesto_rejeita_macroclasse_desconhecida_e_id_duplicado():
    df = make_manifest(IMAGES)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "manifest.csv.gz"

        df_invalido = df.copy()
        df_invalido.loc[0, "Macro_Estilo"] = "Desconhecida"
        df_invalido.to_csv(p, index=False)

        try:
            load_patch_manifest(p)
            raise AssertionError("deveria rejeitar macroclasse desconhecida")
        except ValueError:
            pass

        df_invalido = df.copy()
        df_invalido.loc[1, "Patch_ID"] = df_invalido.loc[0, "Patch_ID"]
        df_invalido.to_csv(p, index=False)

        try:
            load_patch_manifest(p)
            raise AssertionError("deveria rejeitar Patch_ID duplicado")
        except ValueError:
            pass


def test_build_loaders_rejeita_parametros_invalidos():
    with tempfile.TemporaryDirectory() as d:
        for nome in ("train", "validation", "test"):
            df = make_manifest(IMAGES, nome)
            df.to_csv(Path(d) / f"patches_{nome}.csv.gz", index=False)

        for batch_size, k in ((0, 2), (8, 0), (7, 2)):
            try:
                build_loaders(
                    d,
                    SOURCE,
                    batch_size=batch_size,
                    patches_per_painting=k,
                    num_workers=0,
                )
                raise AssertionError(
                    f"deveria rejeitar batch_size={batch_size}, k={k}"
                )
            except ValueError:
                pass

if __name__ == "__main__":
    testes = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in testes:
        t()
        print("OK ", t.__name__)
    print(f"\n{len(testes)} testes passaram")
