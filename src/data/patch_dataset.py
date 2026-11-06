"""Etapa 6 - Dataset, amostrador e DataLoaders dos patches.

Fluxo de um item:
    manifesto de patches (Painting_ID, Indice_Dataset, x0, y0, x1, y1, ...)
      -> busca a pintura no WikiArt pelo Indice_Dataset
      -> recorta imagem[y0:y1, x0:x1]   (nenhum patch é salvo em disco)
      -> augmentation (SOMENTE no treino)
      -> normalização -> float32 CHW

Amostragem do treino (PaintingGroupedSampler):
    Pinturas maiores geram mais patches (Etapa 5, seção 9). Para que nenhuma pintura
    pese mais só por ser grande, cada época sorteia o MESMO número de patches de
    CADA pintura. Como as macroclasses têm o mesmo número de pinturas, as classes
    também ficam balanceadas. Os patches de uma mesma pintura saem em sequência, o
    que permite reaproveitar a imagem já decodificada (cache) e acelera o carregamento.

Validação e teste usam TODOS os patches, sem sorteio e sem augmentation.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .normalization import IMAGENET_MEAN, IMAGENET_STD, normalize_patch

try:  # torch é opcional para importar o módulo (facilita testes)
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = object

PATCH_SIZE = 256
from .labels import CLASSES_FINAIS

CLASS_NAMES = tuple(CLASSES_FINAIS)

CLASS_TO_IDX = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}

REQUIRED_COLUMNS = [
    "Patch_ID", "Painting_ID", "Indice_Dataset", "Split",
    "Macro_Estilo", "Estilo", "x0", "y0", "x1", "y1",
]


# --------------------------------------------------------------------------- manifesto
def load_patch_manifest(path: str | Path) -> pd.DataFrame:
    """Lê um manifesto de patches (.csv ou .csv.gz) e valida o básico."""
    try:
        df = pd.read_csv(path, usecols=REQUIRED_COLUMNS)
    except ValueError as e:
        raise ValueError(f"{path}: colunas esperadas {REQUIRED_COLUMNS}. Detalhe: {e}") from e
    if df["Split"].nunique() != 1:
        raise ValueError(f"{path}: o manifesto deve ter um único Split, achei {df['Split'].unique()}")
    desconhecidas = set(df["Macro_Estilo"]) - set(CLASS_TO_IDX)
    if desconhecidas:
        raise ValueError(f"{path}: macroclasses desconhecidas {desconhecidas}")
    if not df["Patch_ID"].is_unique:
        raise ValueError(f"{path}: Patch_ID repetido")
    return df


# --------------------------------------------------------------------------- imagens
class WikiArtImageSource:
    """Devolve a pintura (uint8 RGB, HWC) a partir do Indice_Dataset.

    O dataset é carregado de forma preguiçosa, dentro de cada processo/worker.
    """

    def __init__(self, hf_name: str = "huggan/wikiart", split: str = "train"):
        self.hf_name, self.split, self._ds = hf_name, split, None

    def __call__(self, indice: int) -> np.ndarray:
        if self._ds is None:
            from datasets import load_dataset
            self._ds = load_dataset(
                self.hf_name,
                split=self.split,
                keep_in_memory=False,
            )
        return np.asarray(self._ds[int(indice)]["image"].convert("RGB"))
# --------------------------------------------------------------------------- augmentation
def augment_treino(patch: np.ndarray) -> np.ndarray:
    """Adaptador para src/data/augmentations.py (Etapa 5).

    A função do projeto, aplicar_augmentation(patch, split, rng), recebe o nome do split;
    o PatchDataset espera uma função que receba só o patch. Aqui o split é fixado em
    "train" (flip horizontal 50% + rotação de até ±10° com 50%). Sem rng, a função usa
    entropia do sistema, então os workers do DataLoader não repetem sorteios.
    """
    from .augmentations import aplicar_augmentation
    return aplicar_augmentation(patch, "train", rng=np.random,)


# --------------------------------------------------------------------------- dataset
class PatchDataset(Dataset):
    """Dataset de patches recortados na hora a partir das pinturas."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_source: Callable[[int], np.ndarray],
        *,
        is_train: bool,
        augment: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        to_tensor: bool = True,
        cache_size: int = 8,
    ):
        if augment is not None and not is_train:
            raise ValueError("augmentation só pode ser usada no treino")
        df = manifest.reset_index(drop=True)
        self.patch_id = df["Patch_ID"].to_numpy()
        self.painting_id = df["Painting_ID"].to_numpy()
        self.indice = df["Indice_Dataset"].to_numpy(np.int64)
        self.x0, self.y0 = df["x0"].to_numpy(np.int64), df["y0"].to_numpy(np.int64)
        self.x1, self.y1 = df["x1"].to_numpy(np.int64), df["y1"].to_numpy(np.int64)
        self.labels = df["Macro_Estilo"].map(CLASS_TO_IDX).to_numpy(np.int64)
        self.image_source, self.augment = image_source, augment
        self.mean, self.std, self.to_tensor = mean, std, to_tensor
        self.cache_size, self._cache = cache_size, OrderedDict()

    def __len__(self) -> int:
        return len(self.patch_id)

    def _get_image(self, indice: int) -> np.ndarray:
        if indice in self._cache:
            self._cache.move_to_end(indice)
            return self._cache[indice]
        img = self.image_source(indice)
        self._cache[indice] = img
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return img

    def get_raw_patch(self, i: int) -> np.ndarray:
        """Patch recortado, sem augmentation e sem normalização (uint8 HWC)."""
        img = self._get_image(int(self.indice[i]))
        x0, y0, x1, y1 = int(self.x0[i]), int(self.y0[i]), int(self.x1[i]), int(self.y1[i])
        h, w = img.shape[:2]
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            raise ValueError(f"{self.patch_id[i]}: patch ({x0},{y0},{x1},{y1}) fora da imagem {w}x{h}")
        patch = np.ascontiguousarray(img[y0:y1, x0:x1])
        if patch.shape != (PATCH_SIZE, PATCH_SIZE, 3):
            raise ValueError(f"{self.patch_id[i]}: patch com formato {patch.shape}")
        return patch

    def __getitem__(self, i: int) -> dict:
        patch = self.get_raw_patch(i)
        if self.augment is not None:
            patch = self.augment(patch)
        arr = normalize_patch(patch, self.mean, self.std)
        if self.to_tensor:
            import torch
            arr = torch.from_numpy(arr)
        return {
            "image": arr,
            "label": int(self.labels[i]),
            "patch_id": str(self.patch_id[i]),
            "painting_id": str(self.painting_id[i]),
            "x0": int(self.x0[i]),
            "y0": int(self.y0[i]),
        }


# --------------------------------------------------------------------------- amostrador
class PaintingGroupedSampler:
    """Sorteia k patches de cada pintura por época, mantendo os patches da mesma pintura juntos.

    - Toda pintura contribui com o mesmo número de patches, independentemente do tamanho.
    - A ordem das pinturas e os patches sorteados mudam a cada época (semente + nº da época).
    - Use batch_size múltiplo de k para que cada lote contenha pinturas completas.
    """

    def __init__(self, painting_ids, patches_per_painting: int = 8, seed: int = 42):
        codes, _ = pd.factorize(np.asarray(painting_ids))
        order = np.argsort(codes, kind="stable")
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        self.groups = np.split(order, bounds)
        self.k, self.seed, self.epoch = patches_per_painting, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return sum(min(self.k, len(g)) for g in self.groups)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1  # a próxima época usa outro sorteio automaticamente
        for p in rng.permutation(len(self.groups)):
            g = self.groups[p]
            yield from rng.choice(g, size=min(self.k, len(g)), replace=False).tolist()


# --------------------------------------------------------------------------- loaders
def _seed_worker(_worker_id: int) -> None:
    import random
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_loaders(
    manifest_dir: str | Path,
    image_source: Callable[[int], np.ndarray],
    *,
    use_augmentation: bool = True,
    augment: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    batch_size: int = 32,
    patches_per_painting: int = 8,
    num_workers: int = 4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    seed: int = 42,
) -> dict:
    """Cria os DataLoaders 'train', 'validation' e 'test'.

    use_augmentation=True usa augment_treino (Etapa 5) no treino; passe False para o
    experimento de ablação sem augmentation. Validação e teste nunca recebem augmentation.
    """
    from torch.utils.data import DataLoader

    import torch

    # Necessário também quando num_workers=0,
    # pois nesse caso _seed_worker() não é executado.
    np.random.seed(seed)

    if not use_augmentation:
        augment = None
    elif augment is None:
        augment = augment_treino

    if batch_size <= 0:
        raise ValueError("batch_size deve ser maior que zero")

    if patches_per_painting <= 0:
        raise ValueError("patches_per_painting deve ser maior que zero")

    if batch_size % patches_per_painting != 0:
        raise ValueError("batch_size deve ser múltiplo de patches_per_painting")
    manifest_dir = Path(manifest_dir)
    common = dict(num_workers=num_workers, worker_init_fn=_seed_worker,
                  persistent_workers=num_workers > 0)

    generator_train = torch.Generator()
    generator_train.manual_seed(seed)

    generator_validation = torch.Generator()
    generator_validation.manual_seed(seed + 1)

    generator_test = torch.Generator()
    generator_test.manual_seed(seed + 2)

    loaders = {}

    train_df = load_patch_manifest(manifest_dir / "patches_train.csv.gz")
    train_ds = PatchDataset(train_df, image_source, is_train=True, augment=augment, mean=mean, std=std)
    sampler = PaintingGroupedSampler(train_ds.painting_id, patches_per_painting, seed)
    loaders["train"] = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, generator=generator_train,**common)

    for name in ("validation", "test"):
        df = load_patch_manifest(manifest_dir / f"patches_{name}.csv.gz")
        ds = PatchDataset(df, image_source, is_train=False, mean=mean, std=std)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, **common)
    return loaders
