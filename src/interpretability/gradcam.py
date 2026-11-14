"""Etapa 11 - Grad-CAM para CNNs (ResNet-50).

Grad-CAM (Selvaraju et al., 2017): para a classe-alvo c,
    alpha_k = média espacial do gradiente de y_c (logit, antes do softmax) em relação ao canal k
    CAM     = ReLU( soma_k alpha_k * A_k )
onde A é a saída da camada convolucional alvo (na ResNet-50, `layer4`, 2048 canais, 8x8
para patches 256x256). O resultado bruto (sem normalizar) é devolvido; a normalização é
feita depois, no nível da pintura, para que patches de uma mesma pintura sejam comparáveis.

Uso: python -m src.interpretability.gradcam   (teste estrutural, sem baixar pesos)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GradCAM:
    def __init__(self, modelo: nn.Module, camada_alvo: nn.Module):
        self.modelo = modelo
        self._ativacao = None
        self._hook = camada_alvo.register_forward_hook(self._guardar)

    def _guardar(self, _modulo, _entrada, saida):
        self._ativacao = saida

    def remover(self) -> None:
        self._hook.remove()

    def __call__(self, x: torch.Tensor, alvo) -> tuple[np.ndarray, torch.Tensor]:
        """x: (B,3,H,W) já normalizado, no dispositivo do modelo. alvo: int ou tensor (B,).

        Retorna (cams (B,h,w) float32 >= 0, logits (B,C) sem gradiente).
        """
        self.modelo.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.modelo(x)
            if isinstance(alvo, int):
                idx = torch.full((x.size(0),), alvo, dtype=torch.long, device=logits.device)
            else:
                idx = alvo.to(logits.device).long()
            score = logits.gather(1, idx[:, None]).sum()
            ativ = self._ativacao
            grad = torch.autograd.grad(score, ativ)[0]
        pesos = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((pesos * ativ).sum(dim=1))
        return cam.detach().float().cpu().numpy(), logits.detach()


def camada_alvo_resnet(modelo: nn.Module) -> nn.Module:
    """Último bloco convolucional da ResNet-50."""
    return modelo.layer4


def testar_gradcam() -> None:
    """Teste estrutural com ResNet-50 sem pesos pré-treinados (não baixa nada)."""
    from src.models.resnet import criar_resnet50

    torch.manual_seed(0)
    modelo = criar_resnet50(usar_pesos_pre_treinados=False, congelar_backbone_inicialmente=False).eval()
    cam = GradCAM(modelo, camada_alvo_resnet(modelo))
    x = torch.randn(3, 3, 256, 256)
    mapas, logits = cam(x, 1)
    assert mapas.shape == (3, 8, 8), mapas.shape
    assert logits.shape == (3, 4)
    assert (mapas >= 0).all()
    assert np.isfinite(mapas).all()
    # alvos diferentes devem gerar mapas diferentes
    mapas2, _ = cam(x, 2)
    assert not np.allclose(mapas, mapas2)
    # logits não dependem do alvo
    with torch.no_grad():
        ref = modelo(x)
    assert torch.allclose(ref, logits, atol=1e-4)
    cam.remover()
    print("Grad-CAM estrutural OK:", mapas.shape, "max", float(mapas.max()))


if __name__ == "__main__":
    testar_gradcam()
