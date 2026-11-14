
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from src.interpretability.mapas import (
    normalizar_mapa,
    reconstruir_mapa,
)


def correlacao_pearson_espacial(a, b):
    """Correlação de Pearson entre dois mapas espaciais."""

    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()

    x = x - x.mean()
    y = y - y.mean()

    denominador = np.sqrt(
        np.dot(x, x) * np.dot(y, y)
    )

    if denominador <= 1e-12:
        return np.nan

    return float(
        np.dot(x, y) / denominador
    )


def mascara_top_fracao(mapa, fracao=0.20):
    """
    Seleciona exatamente a fração de pixels com
    maiores ativações.
    """

    valores = np.asarray(mapa).ravel()

    k = int(round(
        fracao * valores.size
    ))

    k = max(
        1,
        min(k, valores.size),
    )

    indices = np.argpartition(
        valores,
        -k,
    )[-k:]

    mascara = np.zeros(
        valores.size,
        dtype=bool,
    )

    mascara[indices] = True

    return mascara.reshape(
        mapa.shape
    )


def jaccard_top20(a, b):
    """Jaccard entre os 20% de pixels mais ativados."""

    ma = mascara_top_fracao(
        a,
        0.20,
    )

    mb = mascara_top_fracao(
        b,
        0.20,
    )

    intersecao = np.logical_and(
        ma,
        mb,
    ).sum()

    uniao = np.logical_or(
        ma,
        mb,
    ).sum()

    if uniao == 0:
        return np.nan

    return float(
        intersecao / uniao
    )


def main():

    parser = ArgumentParser(
        description=(
            "Compara, nos erros de classificação, "
            "o Grad-CAM direcionado à classe prevista "
            "com o Grad-CAM direcionado à classe real."
        )
    )

    parser.add_argument(
        "--in",
        dest="entrada",
        type=Path,
        default=Path(
            "outputs/gradcam/resnet50_amplo"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    entrada = args.entrada

    if args.out is None:
        saida = (
            entrada
            / "comparacao_numerica_gradcam_erros.csv"
        )
    else:
        saida = args.out

    selecao = pd.read_csv(
        entrada / "selecao.csv"
    )

    erros = selecao[
        selecao["pred_mean_idx"]
        != selecao["true_idx"]
    ].copy()

    cam_dir = entrada / "cams"

    registros = []

    for i, linha in enumerate(
        erros.itertuples(index=False),
        start=1,
    ):

        arquivo = (
            cam_dir
            / f"{linha.Painting_ID}.npz"
        )

        dados = np.load(
            arquivo,
            allow_pickle=False,
        )

        mapa_pred = reconstruir_mapa(
            dados["x0"],
            dados["y0"],
            dados["cams_pred"],
            int(dados["altura"]),
            int(dados["largura"]),
        )

        mapa_real = reconstruir_mapa(
            dados["x0"],
            dados["y0"],
            dados["cams_true"],
            int(dados["altura"]),
            int(dados["largura"]),
        )

        # A comparação utiliza os mesmos mapas
        # normalizados empregados na visualização.
        mapa_pred = normalizar_mapa(
            mapa_pred
        )

        mapa_real = normalizar_mapa(
            mapa_real
        )

        registros.append(
            {
                "Painting_ID":
                    linha.Painting_ID,

                "Indice_Dataset":
                    int(linha.Indice_Dataset),

                "Classe_real":
                    linha.true_class,

                "Classe_prevista":
                    linha.pred_mean_class,

                "Correlacao_espacial":
                    correlacao_pearson_espacial(
                        mapa_pred,
                        mapa_real,
                    ),

                "Jaccard_top20":
                    jaccard_top20(
                        mapa_pred,
                        mapa_real,
                    ),
            }
        )

        if (
            i % 10 == 0
            or i == len(erros)
        ):
            print(
                f"{i}/{len(erros)} pinturas"
            )

    resultado = pd.DataFrame(
        registros
    )

    resultado.to_csv(
        saida,
        index=False,
    )

    print(
        "\nComparação global:"
    )

    for coluna in [
        "Correlacao_espacial",
        "Jaccard_top20",
    ]:

        valores = resultado[coluna]

        print(
            f"\n{coluna}"
        )

        print(
            f"média:   {valores.mean():.4f}"
        )

        print(
            f"mediana: {valores.median():.4f}"
        )

        print(
            f"DP:      {valores.std():.4f}"
        )

    print(
        "\nPor par de confusão:"
    )

    pares = (
        resultado
        .groupby(
            [
                "Classe_real",
                "Classe_prevista",
            ]
        )
        .agg(
            n=("Painting_ID", "size"),
            mediana_correlacao=(
                "Correlacao_espacial",
                "median",
            ),
            mediana_jaccard=(
                "Jaccard_top20",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "Classe_real",
                "Classe_prevista",
            ]
        )
    )

    print(
        pares.to_string(
            index=False
        )
    )

    print(
        f"\nArquivo salvo em: {saida}"
    )


if __name__ == "__main__":
    main()
