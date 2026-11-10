from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.labels import CLASSES_FINAIS
from src.evaluation.evaluate_model import (
    PROB_COLUMNS,
    calcular_metricas,
    salvar_matriz_confusao,
)


def validar_predicoes(df: pd.DataFrame) -> None:
    colunas_necessarias = {
        "Patch_ID",
        "Painting_ID",
        "true_idx",
        "pred_idx",
        "true_class",
        "pred_class",
        *PROB_COLUMNS,
    }

    faltantes = (
        colunas_necessarias
        - set(df.columns)
    )

    if faltantes:
        raise ValueError(
            f"Colunas ausentes: {sorted(faltantes)}"
        )

    if len(df) != 209030:
        raise ValueError(
            f"Esperados 209.030 patches de teste; "
            f"encontrados {len(df)}."
        )

    pinturas = df["Painting_ID"].nunique()

    if pinturas != 1500:
        raise ValueError(
            f"Esperadas 1.500 pinturas; "
            f"encontradas {pinturas}."
        )

    inconsistencias = (
        df.groupby("Painting_ID")["true_idx"]
        .nunique()
        .gt(1)
        .sum()
    )

    if inconsistencias != 0:
        raise ValueError(
            f"{inconsistencias} pinturas possuem "
            "mais de um rótulo real."
        )

    probabilidades = (
        df[PROB_COLUMNS]
        .sum(axis=1)
        .to_numpy()
    )

    if not np.allclose(
        probabilidades,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Existem linhas cujas probabilidades "
            "não somam aproximadamente 1."
        )


def agregar_pinturas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    registros = []

    numero_classes = len(
        CLASSES_FINAIS
    )

    for painting_id, grupo in df.groupby(
        "Painting_ID",
        sort=False,
    ):
        true_idx = int(
            grupo["true_idx"].iloc[0]
        )

        true_class = (
            CLASSES_FINAIS[true_idx]
        )

        votos = np.bincount(
            grupo["pred_idx"].to_numpy(
                dtype=np.int64
            ),
            minlength=numero_classes,
        )

        maior_numero_votos = (
            votos.max()
        )

        classes_empatadas = np.flatnonzero(
            votos == maior_numero_votos
        )

        probabilidades_medias = (
            grupo[PROB_COLUMNS]
            .mean(axis=0)
            .to_numpy(
                dtype=np.float64
            )
        )

        houve_empate = (
            len(classes_empatadas) > 1
        )

        if houve_empate:
            probabilidades_empatadas = (
                probabilidades_medias[
                    classes_empatadas
                ]
            )

            indice_local = int(
                np.argmax(
                    probabilidades_empatadas
                )
            )

            pred_majority_idx = int(
                classes_empatadas[
                    indice_local
                ]
            )

        else:
            pred_majority_idx = int(
                classes_empatadas[0]
            )

        pred_mean_idx = int(
            np.argmax(
                probabilidades_medias
            )
        )

        registro = {
            "Painting_ID": painting_id,
            "true_idx": true_idx,
            "true_class": true_class,
            "n_patches": int(
                len(grupo)
            ),
            "majority_tie": bool(
                houve_empate
            ),
            "pred_majority_idx":
                pred_majority_idx,
            "pred_majority_class":
                CLASSES_FINAIS[
                    pred_majority_idx
                ],
            "pred_mean_idx":
                pred_mean_idx,
            "pred_mean_class":
                CLASSES_FINAIS[
                    pred_mean_idx
                ],
        }

        for class_idx, classe in enumerate(
            CLASSES_FINAIS
        ):
            registro[
                f"votes_{classe}"
            ] = int(
                votos[class_idx]
            )

            registro[
                f"mean_prob_{classe}"
            ] = float(
                probabilidades_medias[
                    class_idx
                ]
            )

        registros.append(
            registro
        )

    resultado = pd.DataFrame(
        registros
    )

    if len(resultado) != 1500:
        raise AssertionError(
            "A agregação não gerou exatamente "
            "1.500 pinturas."
        )

    return resultado


def salvar_resultados_metodo(
    df_pinturas: pd.DataFrame,
    *,
    model_name: str,
    method: str,
    pred_column: str,
    output_dir: Path,
) -> dict:
    resumo, por_classe = (
        calcular_metricas(
            df_pinturas["true_idx"],
            df_pinturas[pred_column],
        )
    )

    resumo.update(
        {
            "model": model_name,
            "split": "test",
            "level": "painting",
            "aggregation": method,
        }
    )

    summary_path = (
        output_dir
        / f"test_painting_{method}_summary.json"
    )

    metrics_path = (
        output_dir
        / (
            f"test_painting_{method}"
            "_metrics_by_class.csv"
        )
    )

    confusion_path = (
        output_dir
        / (
            f"test_painting_{method}"
            "_confusion.png"
        )
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            resumo,
            arquivo,
            indent=2,
            ensure_ascii=False,
        )

    por_classe.to_csv(
        metrics_path,
        index=False,
    )

    salvar_matriz_confusao(
        df_pinturas["true_idx"],
        df_pinturas[pred_column],
        confusion_path,
        (
            f"{model_name} — teste — "
            f"pinturas — {method}"
        ),
    )

    return resumo


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet50",
            "vit_b16",
        ],
    )

    parser.add_argument(
        "--predictions",
        required=True,
    )

    args = parser.parse_args()

    predictions_path = Path(
        args.predictions
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            predictions_path
        )

    print(
        "Carregando:",
        predictions_path,
    )

    df = pd.read_csv(
        predictions_path
    )

    validar_predicoes(
        df
    )

    print(
        "Patches validados:",
        len(df),
    )

    print(
        "Pinturas:",
        df["Painting_ID"].nunique(),
    )

    df_pinturas = agregar_pinturas(
        df
    )

    output_dir = (
        Path("outputs")
        / "evaluation"
        / args.model
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_output = (
        output_dir
        / "test_painting_predictions.csv.gz"
    )

    df_pinturas.to_csv(
        predictions_output,
        index=False,
        compression="gzip",
    )

    majority_summary = (
        salvar_resultados_metodo(
            df_pinturas,
            model_name=args.model,
            method="majority",
            pred_column="pred_majority_idx",
            output_dir=output_dir,
        )
    )

    mean_summary = (
        salvar_resultados_metodo(
            df_pinturas,
            model_name=args.model,
            method="mean_probability",
            pred_column="pred_mean_idx",
            output_dir=output_dir,
        )
    )

    numero_empates = int(
        df_pinturas[
            "majority_tie"
        ].sum()
    )

    print()
    print(
        "=== Agregação concluída ==="
    )

    print(
        "Pinturas avaliadas:",
        len(df_pinturas),
    )

    print(
        "Empates na votação majoritária:",
        numero_empates,
    )

    print()
    print(
        "VOTAÇÃO MAJORITÁRIA"
    )

    print(
        json.dumps(
            majority_summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "MÉDIA DAS PROBABILIDADES"
    )

    print(
        json.dumps(
            mean_summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "Predições por pintura:",
        predictions_output,
    )


if __name__ == "__main__":
    main()