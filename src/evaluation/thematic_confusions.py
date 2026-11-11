from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.labels import CLASSES_FINAIS
from src.evaluation.shortcut_analysis import (
    carregar_metadata,
    carregar_predicoes,
    preparar_base_teste,
)


FOCUS_PAIRS = [
    (
        "Renaissance",
        "Baroque",
    ),
    (
        "Baroque",
        "Renaissance",
    ),
]

FOCUS_GENRES = [
    "religious_painting",
    "portrait",
]


def calcular_confusoes_por_genero(
    base: pd.DataFrame,
    *,
    model_name: str,
    pred_column: str,
) -> pd.DataFrame:

    registros = []

    for classe_real in CLASSES_FINAIS:

        base_classe = base[
            base["Macro_Estilo"]
            == classe_real
        ]

        for classe_predita in CLASSES_FINAIS:

            if (
                classe_predita
                == classe_real
            ):
                continue

            total_erros_par = int(
                (
                    base_classe[
                        pred_column
                    ]
                    == classe_predita
                ).sum()
            )

            for (
                genero,
                grupo,
            ) in base_classe.groupby(
                "Genero",
                sort=True,
            ):

                n_origem = len(
                    grupo
                )

                n_confusoes = int(
                    (
                        grupo[
                            pred_column
                        ]
                        == classe_predita
                    ).sum()
                )

                taxa = (
                    n_confusoes
                    / n_origem
                    if n_origem > 0
                    else 0.0
                )

                proporcao_erros_par = (
                    n_confusoes
                    / total_erros_par
                    if total_erros_par > 0
                    else 0.0
                )

                registros.append(
                    {
                        "model":
                            model_name,
                        "true_class":
                            classe_real,
                        "predicted_class":
                            classe_predita,
                        "Genero":
                            genero,
                        "n_source_genre":
                            n_origem,
                        "n_confusions":
                            n_confusoes,
                        "confusion_rate":
                            taxa,
                        "pair_errors_total":
                            total_erros_par,
                        "share_of_pair_errors":
                            proporcao_erros_par,
                    }
                )

    return pd.DataFrame(
        registros
    )


def criar_resumo_foco(
    base: pd.DataFrame,
    *,
    model_name: str,
    pred_column: str,
) -> pd.DataFrame:

    registros = []

    for (
        classe_real,
        classe_predita,
    ) in FOCUS_PAIRS:

        origem = base[
            base["Macro_Estilo"]
            == classe_real
        ]

        total_origem = len(
            origem
        )

        total_confusoes = int(
            (
                origem[
                    pred_column
                ]
                == classe_predita
            ).sum()
        )

        registros.append(
            {
                "model":
                    model_name,
                "direction":
                    (
                        f"{classe_real}"
                        f" → "
                        f"{classe_predita}"
                    ),
                "scope":
                    "full_test",
                "n_source":
                    total_origem,
                "n_confusions":
                    total_confusoes,
                "confusion_rate":
                    (
                        total_confusoes
                        / total_origem
                    ),
                "share_of_pair_errors":
                    1.0,
            }
        )

        for genero in FOCUS_GENRES:

            grupo = origem[
                origem["Genero"]
                == genero
            ]

            n_origem = len(
                grupo
            )

            n_confusoes = int(
                (
                    grupo[
                        pred_column
                    ]
                    == classe_predita
                ).sum()
            )

            registros.append(
                {
                    "model":
                        model_name,
                    "direction":
                        (
                            f"{classe_real}"
                            f" → "
                            f"{classe_predita}"
                        ),
                    "scope":
                        genero,
                    "n_source":
                        n_origem,
                    "n_confusions":
                        n_confusoes,
                    "confusion_rate":
                        (
                            n_confusoes
                            / n_origem
                            if n_origem > 0
                            else 0.0
                        ),
                    "share_of_pair_errors":
                        (
                            n_confusoes
                            / total_confusoes
                            if total_confusoes > 0
                            else 0.0
                        ),
                }
            )

    return pd.DataFrame(
        registros
    )


def comparar_religious_portrait(
    resumo: pd.DataFrame,
) -> pd.DataFrame:

    registros = []

    for (
        model,
        direction,
    ), grupo in resumo.groupby(
        [
            "model",
            "direction",
        ],
        sort=False,
    ):

        religious = grupo[
            grupo["scope"]
            == "religious_painting"
        ]

        portrait = grupo[
            grupo["scope"]
            == "portrait"
        ]

        full = grupo[
            grupo["scope"]
            == "full_test"
        ]

        if (
            religious.empty
            or portrait.empty
            or full.empty
        ):
            continue

        religious_row = (
            religious.iloc[0]
        )

        portrait_row = (
            portrait.iloc[0]
        )

        full_row = (
            full.iloc[0]
        )

        registros.append(
            {
                "model":
                    model,
                "direction":
                    direction,
                "full_rate":
                    full_row[
                        "confusion_rate"
                    ],
                "religious_rate":
                    religious_row[
                        "confusion_rate"
                    ],
                "portrait_rate":
                    portrait_row[
                        "confusion_rate"
                    ],
                "delta_religious_vs_portrait":
                    (
                        religious_row[
                            "confusion_rate"
                        ]
                        - portrait_row[
                            "confusion_rate"
                        ]
                    ),
                "religious_n":
                    int(
                        religious_row[
                            "n_source"
                        ]
                    ),
                "religious_errors":
                    int(
                        religious_row[
                            "n_confusions"
                        ]
                    ),
                "portrait_n":
                    int(
                        portrait_row[
                            "n_source"
                        ]
                    ),
                "portrait_errors":
                    int(
                        portrait_row[
                            "n_confusions"
                        ]
                    ),
                "religious_share_of_pair_errors":
                    religious_row[
                        "share_of_pair_errors"
                    ],
                "portrait_share_of_pair_errors":
                    portrait_row[
                        "share_of_pair_errors"
                    ],
            }
        )

    return pd.DataFrame(
        registros
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        default=(
            "data/metadata/test.csv"
        ),
    )

    parser.add_argument(
        "--resnet-predictions",
        default=(
            "outputs/evaluation/resnet50/"
            "test_painting_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--vit-predictions",
        default=(
            "outputs/evaluation/vit_b16/"
            "test_painting_predictions.csv.gz"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/shortcut_learning"
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_df = carregar_metadata(
        Path(args.test)
    )

    resnet_df = carregar_predicoes(
        Path(
            args.resnet_predictions
        ),
        "resnet50",
    )

    vit_df = carregar_predicoes(
        Path(
            args.vit_predictions
        ),
        "vit_b16",
    )

    base = preparar_base_teste(
        test_df,
        resnet_df,
        vit_df,
    )

    configuracoes = [
        (
            "ResNet-50",
            "pred_mean_resnet50",
        ),
        (
            "ViT-B/16",
            "pred_mean_vit_b16",
        ),
    ]

    todas_confusoes = []

    resumos = []

    for (
        model_name,
        pred_column,
    ) in configuracoes:

        todas_confusoes.append(
            calcular_confusoes_por_genero(
                base,
                model_name=model_name,
                pred_column=pred_column,
            )
        )

        resumos.append(
            criar_resumo_foco(
                base,
                model_name=model_name,
                pred_column=pred_column,
            )
        )

    todas_confusoes_df = (
        pd.concat(
            todas_confusoes,
            ignore_index=True,
        )
    )

    resumo_df = pd.concat(
        resumos,
        ignore_index=True,
    )

    comparacao_df = (
        comparar_religious_portrait(
            resumo_df
        )
    )

    todas_confusoes_df.to_csv(
        output_dir
        / "thematic_confusions_all.csv",
        index=False,
    )

    resumo_df.to_csv(
        output_dir
        / "renaissance_baroque_confusions.csv",
        index=False,
    )

    comparacao_df.to_csv(
        output_dir
        / (
            "renaissance_baroque_"
            "religious_vs_portrait.csv"
        ),
        index=False,
    )

    print()
    print(
        "=== Etapa 10E — "
        "confusões temáticas ==="
    )

    print()
    print(
        "=== Renaissance ↔ Baroque ==="
    )

    tabela = resumo_df.copy()

    tabela[
        "confusion_rate"
    ] *= 100

    tabela[
        "share_of_pair_errors"
    ] *= 100

    print(
        tabela.round(2)
        .to_string(index=False)
    )

    print()
    print(
        "=== Religious painting "
        "vs portrait ==="
    )

    tabela_comparacao = (
        comparacao_df.copy()
    )

    for coluna in [
        "full_rate",
        "religious_rate",
        "portrait_rate",
        "delta_religious_vs_portrait",
        "religious_share_of_pair_errors",
        "portrait_share_of_pair_errors",
    ]:
        tabela_comparacao[
            coluna
        ] *= 100

    print(
        tabela_comparacao
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()