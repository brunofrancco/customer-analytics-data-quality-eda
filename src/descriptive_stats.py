"""
Módulo de Estatística Descritiva.

Calcula um conjunto completo de métricas descritivas para as variáveis
numéricas de interesse do negócio, incluindo medidas de tendência central,
dispersão, forma da distribuição e percentis detalhados.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

COLUNAS_NUMERICAS_PADRAO = [
    "age",
    "income",
    "purchase_count",
    "total_spent",
    "satisfaction_score",
]

PERCENTIS_DETALHADOS = [10, 25, 50, 75, 90, 95, 99]


def _moda(serie: pd.Series):
    moda = serie.mode()
    return moda.iloc[0] if not moda.empty else np.nan


def compute_descriptive_statistics(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Calcula estatísticas descritivas completas para as colunas numéricas
    informadas (por padrão, as principais variáveis de negócio do projeto).

    Métricas: count, mean, median, mode, std, variance, min, max, range,
    Q1, Q3, IQR, percentis P10-P99, skewness, kurtosis e coeficiente de
    variação (%).
    """

    colunas = columns or [c for c in COLUNAS_NUMERICAS_PADRAO if c in df.columns]
    linhas = []

    for coluna in colunas:
        serie = pd.to_numeric(df[coluna], errors="coerce").dropna()
        if serie.empty:
            continue

        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        percentis = {f"p{p}": serie.quantile(p / 100) for p in PERCENTIS_DETALHADOS}
        desvio_padrao = serie.std()
        media = serie.mean()

        linha = {
            "column": coluna,
            "count": int(serie.count()),
            "mean": media,
            "median": serie.median(),
            "mode": _moda(serie),
            "std": desvio_padrao,
            "variance": serie.var(),
            "min": serie.min(),
            "max": serie.max(),
            "range": serie.max() - serie.min(),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            **percentis,
            "skewness": float(stats.skew(serie)),
            "kurtosis": float(stats.kurtosis(serie)),
            "coefficient_of_variation": float(desvio_padrao / media * 100) if media else np.nan,
        }
        linhas.append(linha)

    resultado = pd.DataFrame(linhas).set_index("column")
    logger.info("Descriptive statistics computed for columns: %s", colunas)
    return resultado


def classify_distribution_shape(skewness: float, kurtosis: float) -> dict[str, str]:
    """
    Classifica a forma de uma distribuição a partir de assimetria (skewness)
    e curtose (kurtosis, definição de excesso — 0 para a normal).
    """

    if skewness > 1:
        forma_assimetria = "fortemente assimétrica à direita (cauda longa de valores altos)"
    elif skewness > 0.5:
        forma_assimetria = "moderadamente assimétrica à direita"
    elif skewness < -1:
        forma_assimetria = "fortemente assimétrica à esquerda (cauda longa de valores baixos)"
    elif skewness < -0.5:
        forma_assimetria = "moderadamente assimétrica à esquerda"
    else:
        forma_assimetria = "aproximadamente simétrica"

    if kurtosis > 1:
        forma_curtose = "leptocúrtica (mais concentrada, com caudas mais pesadas que a normal)"
    elif kurtosis < -1:
        forma_curtose = "platicúrtica (mais achatada que a normal, caudas mais leves)"
    else:
        forma_curtose = "mesocúrtica (próxima da distribuição normal)"

    return {"assimetria": forma_assimetria, "curtose": forma_curtose}


# ---------------------------------------------------------------------------
# Detecção de outliers (método IQR)
# ---------------------------------------------------------------------------

def detect_outliers_iqr(
    df: pd.DataFrame, columns: list[str] | None = None, multiplicador: float = 1.5
) -> pd.DataFrame:
    """
    Detecta outliers em variáveis numéricas pelo método do IQR
    (Intervalo Interquartil): valores abaixo de ``Q1 - k*IQR`` ou acima de
    ``Q3 + k*IQR`` (k=1.5 por padrão) são sinalizados como outliers.

    Os outliers NÃO são removidos — apenas identificados e quantificados,
    para posterior decisão de negócio (manter, investigar, transformar ou
    corrigir).
    """

    colunas = columns or [c for c in COLUNAS_NUMERICAS_PADRAO if c in df.columns]
    linhas = []

    for coluna in colunas:
        serie = pd.to_numeric(df[coluna], errors="coerce").dropna()
        if serie.empty:
            continue

        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - multiplicador * iqr
        limite_superior = q3 + multiplicador * iqr

        outliers = serie[(serie < limite_inferior) | (serie > limite_superior)]

        linhas.append(
            {
                "column": coluna,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": limite_inferior,
                "upper_bound": limite_superior,
                "total_records": int(serie.count()),
                "outlier_count": int(outliers.shape[0]),
                "outlier_percentage": round(100 * outliers.shape[0] / serie.shape[0], 2)
                if serie.shape[0]
                else 0.0,
            }
        )

    resultado = pd.DataFrame(linhas).set_index("column")
    logger.info("Outlier detection (IQR) completed for columns: %s", colunas)
    return resultado


def recommend_outlier_action(outlier_percentage: float, business_context: str = "") -> str:
    """
    Sugere uma diretriz geral para tratamento de outliers com base no seu
    percentual de ocorrência. Esta é uma recomendação de ponto de partida —
    a decisão final deve sempre considerar o contexto de negócio.
    """

    if outlier_percentage == 0:
        return "Nenhum outlier detectado — nenhuma ação necessária."
    if outlier_percentage < 1:
        return (
            "Baixa incidência: manter os valores e investigar individualmente "
            "(podem ser clientes legítimos de alto valor)."
        )
    if outlier_percentage < 5:
        return (
            "Incidência moderada: investigar a origem (erro de digitação vs. "
            "cliente atípico legítimo) antes de decidir entre manter, corrigir "
            "ou aplicar transformação (ex.: log) para análises sensíveis a escala."
        )
    return (
        "Alta incidência: reavaliar a regra de negócio ou o processo de coleta — "
        "um percentual tão alto de outliers pode indicar um problema sistemático "
        "de qualidade de dados, não apenas casos isolados."
    )


# ---------------------------------------------------------------------------
# Correlação
# ---------------------------------------------------------------------------

def compute_correlation_matrix(
    df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson"
) -> pd.DataFrame:
    """Calcula a matriz de correlação (Pearson ou Spearman) entre variáveis numéricas."""

    colunas = columns or [c for c in COLUNAS_NUMERICAS_PADRAO if c in df.columns]
    numericas = df[colunas].apply(pd.to_numeric, errors="coerce")
    return numericas.corr(method=method)


def top_correlated_pairs(
    df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson", top_n: int = 10
) -> pd.DataFrame:
    """Retorna os pares de variáveis mais correlacionados (em módulo), ordenados."""

    corr = compute_correlation_matrix(df, columns=columns, method=method)
    pares = []
    colunas_lista = corr.columns.tolist()
    for i, col_a in enumerate(colunas_lista):
        for col_b in colunas_lista[i + 1 :]:
            pares.append(
                {
                    "variable_1": col_a,
                    "variable_2": col_b,
                    "method": method,
                    "correlation": corr.loc[col_a, col_b],
                }
            )
    resultado = pd.DataFrame(pares)
    if resultado.empty:
        return resultado
    resultado["correlation_abs"] = resultado["correlation"].abs()
    resultado = resultado.sort_values("correlation_abs", ascending=False, ignore_index=True)
    return resultado.drop(columns="correlation_abs").head(top_n)


# ---------------------------------------------------------------------------
# Segmentação de clientes
# ---------------------------------------------------------------------------

def create_customer_segments(
    df: pd.DataFrame, column: str = "total_spent", new_column: str = "customer_segment"
) -> pd.DataFrame:
    """
    Cria segmentos de clientes (Low / Medium / High Value) a partir dos
    tercis (percentis 33 e 66) de ``total_spent``. O uso de tercis, em vez
    de limiares arbitrários, garante uma divisão estatisticamente
    justificável e equilibrada em número de clientes por segmento.
    """

    df = df.copy()
    limite_inferior = df[column].quantile(1 / 3)
    limite_superior = df[column].quantile(2 / 3)

    def _classificar(valor: float) -> str:
        if pd.isna(valor):
            return np.nan
        if valor <= limite_inferior:
            return "Low Value"
        if valor <= limite_superior:
            return "Medium Value"
        return "High Value"

    df[new_column] = df[column].apply(_classificar)
    logger.info(
        "Customer segments created: thresholds P33=%.2f, P66=%.2f", limite_inferior, limite_superior
    )
    return df


def summarize_segments(
    df: pd.DataFrame,
    segment_column: str = "customer_segment",
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compara os segmentos de clientes em relação a métricas-chave de negócio."""

    colunas_metricas = metrics or ["age", "income", "purchase_count", "satisfaction_score", "total_spent"]
    colunas_existentes = [c for c in colunas_metricas if c in df.columns]

    resumo = (
        df.groupby(segment_column)[colunas_existentes]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    ordem = ["Low Value", "Medium Value", "High Value"]
    resumo = resumo.reindex([o for o in ordem if o in resumo.index])
    return resumo
