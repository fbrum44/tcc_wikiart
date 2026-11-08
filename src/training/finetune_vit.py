"""Etapa 8 - Fine-tuning do ViT-B/16.

Espelha src/training/finetune_resnet.py: parte do melhor checkpoint da cabeça, libera todo
o modelo, usa AdamW com LR diferenciado (backbone 1e-5, cabeça 1e-4) e salva o melhor
checkpoint por menor validation loss. Reaproveita o laço de treino/validação da ResNet,
que é genérico (o ViT não tem BatchNorm, então o congelamento de estatísticas não age).
"""

from __future__ import annotations

import argparse
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import timm
import torch
from torch import nn

from src.data.labels import CLASSES_FINAIS
from src.data.normalization import HALF_MEAN, HALF_STD
from src.data.patch_dataset import WikiArtImageSource, build_loaders
from src.models.vit import (
    INPUT_SIZE,
    VIT_MODEL_NAME,
    VIT_WEIGHTS,
    criar_vit,
    descongelar_modelo,
)
from src.training.finetune_resnet import (
    salvar_checkpoint,
    salvar_configuracao,
    salvar_historico,
    treinar_epoca,
    validar_epoca,
)
from src.training.train_resnet import configurar_seed, obter_dispositivo
from src.training.train_resnet_head import criar_loader_validacao_fixo


@dataclass(frozen=True)
class FineTuneConfig:
    seed: int = 42

    epochs: int = 3
    batch_size: int = 32

    train_patches_per_painting: int = 8
    val_patches_per_painting: int = 8

    backbone_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4

    weight_decay: float = 1e-4

    num_workers: int = 0

    architecture: str = "vit_base_patch16"
    weights: str = VIT_WEIGHTS
    timm_model: str = VIT_MODEL_NAME

    input_size: int = INPUT_SIZE

    source_checkpoint: str = "models/vit_head_best.pt"


def carregar_modelo_da_cabeca(checkpoint_path: Path) -> tuple[nn.Module, dict]:

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if list(checkpoint.get("classes")) != list(CLASSES_FINAIS):
        raise ValueError("As classes do checkpoint não coincidem com CLASSES_FINAIS.")

    modelo = criar_vit(
        usar_pesos_pre_treinados=False,
        congelar_backbone_inicialmente=False,
    )
    modelo.load_state_dict(checkpoint["model_state_dict"], strict=True)
    descongelar_modelo(modelo)

    return modelo, checkpoint


def criar_otimizador(modelo: nn.Module, config: FineTuneConfig):
    """Backbone e cabeça recebem learning rates diferentes."""

    parametros_backbone = [p for n, p in modelo.named_parameters() if not n.startswith("head.")]
    parametros_head = list(modelo.head.parameters())

    return torch.optim.AdamW(
        [
            {"params": parametros_backbone, "lr": config.backbone_learning_rate},
            {"params": parametros_head, "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )


def criar_registro_configuracao(config, dispositivo, checkpoint_cabeca: dict) -> dict:

    registro = asdict(config)
    registro.update(
        {
            "classes": CLASSES_FINAIS,
            "normalization_mean": list(HALF_MEAN),
            "normalization_std": list(HALF_STD),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "timm_version": timm.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(dispositivo),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "phase": "full_finetuning",
            "validation_strategy": "fixed_patches_per_painting",
            "checkpoint_metric": "validation_loss",
            "source_head_epoch": checkpoint_cabeca.get("epoch"),
            "source_head_val_loss": checkpoint_cabeca.get("val_loss"),
            "source_head_val_accuracy": checkpoint_cabeca.get("val_accuracy"),
        }
    )
    return registro


def executar_finetuning(config: FineTuneConfig, *, smoke: bool = False) -> None:

    print("=== Etapa 8 — Fine-tuning ViT-B/16 ===")

    configurar_seed(config.seed)
    dispositivo = obter_dispositivo()
    print("Dispositivo:", dispositivo)

    source_path = Path(config.source_checkpoint)
    modelo, checkpoint_cabeca = carregar_modelo_da_cabeca(source_path)

    print("Checkpoint inicial:", source_path)
    print("Validation loss da cabeça:", checkpoint_cabeca.get("val_loss"))
    print("Validation accuracy da cabeça:", checkpoint_cabeca.get("val_accuracy"))

    modelo = modelo.to(dispositivo)

    nao_treinaveis = [n for n, p in modelo.named_parameters() if not p.requires_grad]
    if nao_treinaveis:
        raise AssertionError(f"Existem parâmetros congelados no fine-tuning: {nao_treinaveis[:5]}")

    patch_antes = modelo.patch_embed.proj.weight.detach().cpu().clone()
    head_antes = modelo.head.weight.detach().cpu().clone()

    loaders = build_loaders(
        manifest_dir="data/metadata",
        image_source=WikiArtImageSource(),
        use_augmentation=True,
        batch_size=config.batch_size,
        patches_per_painting=config.train_patches_per_painting,
        num_workers=config.num_workers,
        mean=HALF_MEAN,
        std=HALF_STD,
        seed=config.seed,
    )

    val_loader = criar_loader_validacao_fixo(
        loaders["validation"],
        patches_per_painting=config.val_patches_per_painting,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed + 1000,
    )

    print("Patches de treino por época:", len(loaders["train"].sampler))
    print("Patches de validação por época:", len(val_loader.sampler))

    criterio = nn.CrossEntropyLoss()
    otimizador = criar_otimizador(modelo, config)

    print("LR backbone:", config.backbone_learning_rate)
    print("LR head:", config.head_learning_rate)

    sufixo = "_smoke" if smoke else ""
    history_path = Path(f"outputs/vit/finetune{sufixo}_history.csv")
    config_path = Path(f"outputs/vit/finetune{sufixo}_config.json")
    checkpoint_path = Path("models/vit_finetune_smoke.pt" if smoke else "models/vit_finetune_best.pt")

    registro_config = criar_registro_configuracao(config, dispositivo, checkpoint_cabeca)
    registro_config["smoke_test"] = smoke
    salvar_configuracao(registro_config, config_path)

    historico = []
    melhor_val_loss = float("inf")
    melhor_epoca = None

    epochs_executar = 1 if smoke else config.epochs
    head_epoch = int(checkpoint_cabeca.get("epoch", 3))

    for finetune_epoch in range(1, epochs_executar + 1):

        overall_epoch = head_epoch + finetune_epoch

        print()
        print(
            f"===== Fine-tuning {finetune_epoch}/{epochs_executar} "
            f"(época geral {overall_epoch}) ====="
        )

        train_loss, train_acc = treinar_epoca(
            modelo, loaders["train"], criterio, otimizador, dispositivo,
            max_batches=5 if smoke else None,
        )
        val_loss, val_acc = validar_epoca(
            modelo, val_loader, criterio, dispositivo,
            max_batches=3 if smoke else None,
        )

        eh_melhor = val_loss < melhor_val_loss

        historico.append(
            {
                "finetune_epoch": finetune_epoch,
                "overall_epoch": overall_epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "best": eh_melhor,
            }
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

        if eh_melhor:
            melhor_val_loss = val_loss
            melhor_epoca = finetune_epoch

            salvar_checkpoint(
                checkpoint_path,
                modelo,
                otimizador,
                finetune_epoch=finetune_epoch,
                overall_epoch=overall_epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                config=registro_config,
                historico=historico,
            )
            print("Novo melhor checkpoint do fine-tuning salvo.")

        salvar_historico(historico, history_path)

    if torch.equal(patch_antes, modelo.patch_embed.proj.weight.detach().cpu()):
        raise AssertionError("O backbone não foi atualizado durante o fine-tuning.")

    if torch.equal(head_antes, modelo.head.weight.detach().cpu()):
        raise AssertionError("A cabeça não foi atualizada durante o fine-tuning.")

    if not checkpoint_path.exists():
        raise AssertionError("Checkpoint não foi criado.")

    head_val_loss = float(checkpoint_cabeca["val_loss"])

    print()
    print("=== Fine-tuning concluído ===")
    print("Melhor época do fine-tuning:", melhor_epoca)
    print("Melhor validation loss do fine-tuning:", f"{melhor_val_loss:.4f}")
    print("Validation loss da cabeça:", f"{head_val_loss:.4f}")
    print(
        "Fine-tuning superou o checkpoint da cabeça:",
        "SIM" if melhor_val_loss < head_val_loss else "NÃO",
    )
    print("Backbone foi atualizado: SIM")
    print("Cabeça foi atualizada: SIM")
    print("Histórico:", history_path)
    print("Configuração:", config_path)
    print("Melhor checkpoint:", checkpoint_path)


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-k", type=int, default=8)
    parser.add_argument("--val-k", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    executar_finetuning(
        FineTuneConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_patches_per_painting=args.train_k,
            val_patches_per_painting=args.val_k,
            num_workers=args.num_workers,
        ),
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
