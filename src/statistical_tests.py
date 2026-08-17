"""
Módulo de Testes Estatísticos (Estatística Inferencial).

Implementa wrappers sobre ``scipy.stats`` que padronizam a saída de cada
teste em um dicionário com hipótese nula, hipótese alternativa, estatística
do teste, p-valor, alpha e uma interpretação em linguagem de negócio.

Todos os testes utilizam alpha = 0.05, salvo indicação em contrário.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

ALPHA_PADRAO = 0.05


def _linha_resultado(
    nome_teste: str,
    h0: str,
    h1: str,
    estatistica: float,
    p_valor: float,
    alpha: float,
    interpretacao_rejeita: str,
    interpretacao_nao_rejeita: str,
) -> dict:
    rejeita_h0 = p_valor < alpha
    resultado = "H0 rejeitada" if rejeita_h0 else "H0 não rejeitada"
    interpretacao = interpretacao_rejeita if rejeita_h0 else interpretacao_nao_rejeita

    return {
        "test": nome_teste,
        "null_hypothesis": h0,
        "alternative_hypothesis": h1,
        "statistic": round(float(estatistica), 4),
        "p_value": round(float(p_valor), 6),
        "alpha": alpha,
        "result": resultado,
        "interpretation": interpretacao,
    }


def pearson_correlation_test(
    df: pd.DataFrame, column_x: str, column_y: str, alpha: float = ALPHA_PADRAO
) -> dict:
    """Testa correlação linear (Pearson) entre duas variáveis numéricas."""

    dados = df[[column_x, column_y]].apply(pd.to_numeric, errors="coerce").dropna()
    estatistica, p_valor = stats.pearsonr(dados[column_x], dados[column_y])

    return _linha_resultado(
        nome_teste=f"Pearson correlation ({column_x} x {column_y})",
        h0=f"Não há correlação linear entre {column_x} e {column_y} (rho = 0)",
        h1=f"Há correlação linear entre {column_x} e {column_y} (rho != 0)",
        estatistica=estatistica,
        p_valor=p_valor,
        alpha=alpha,
        interpretacao_rejeita=(
            f"Existe evidência estatística de correlação linear entre {column_x} e "
            f"{column_y} (r={estatistica:.3f}). Correlação não implica causalidade."
        ),
        interpretacao_nao_rejeita=(
            f"Não há evidência estatística suficiente de correlação linear entre "
            f"{column_x} e {column_y}."
        ),
    )


def spearman_correlation_test(
    df: pd.DataFrame, column_x: str, column_y: str, alpha: float = ALPHA_PADRAO
) -> dict:
    """Testa correlação monotônica (Spearman) entre duas variáveis numéricas."""

    dados = df[[column_x, column_y]].apply(pd.to_numeric, errors="coerce").dropna()
    estatistica, p_valor = stats.spearmanr(dados[column_x], dados[column_y])

    return _linha_resultado(
        nome_teste=f"Spearman correlation ({column_x} x {column_y})",
        h0=f"Não há correlação monotônica entre {column_x} e {column_y} (rho = 0)",
        h1=f"Há correlação monotônica entre {column_x} e {column_y} (rho != 0)",
        estatistica=estatistica,
        p_valor=p_valor,
        alpha=alpha,
        interpretacao_rejeita=(
            f"Existe evidência estatística de correlação monotônica entre {column_x} "
            f"e {column_y} (rho={estatistica:.3f})."
        ),
        interpretacao_nao_rejeita=(
            f"Não há evidência estatística suficiente de correlação monotônica entre "
            f"{column_x} e {column_y}."
        ),
    )


def independent_t_test(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    group_a,
    group_b,
    alpha: float = ALPHA_PADRAO,
) -> dict:
    """
    Testa se as médias de ``value_column`` diferem entre dois grupos
    independentes definidos por ``group_column``. Assume variâncias
    desiguais por padrão (Welch's t-test), mais robusto quando essa
    premissa não pode ser garantida.
    """

    dados = df[[value_column, group_column]].dropna()
    amostra_a = pd.to_numeric(dados.loc[dados[group_column] == group_a, value_column], errors="coerce").dropna()
    amostra_b = pd.to_numeric(dados.loc[dados[group_column] == group_b, value_column], errors="coerce").dropna()

    estatistica, p_valor = stats.ttest_ind(amostra_a, amostra_b, equal_var=False)

    return _linha_resultado(
        nome_teste=f"Independent T-Test ({value_column}: {group_a} vs {group_b})",
        h0=f"A média de {value_column} é igual entre os grupos '{group_a}' e '{group_b}'",
        h1=f"A média de {value_column} é diferente entre os grupos '{group_a}' e '{group_b}'",
        estatistica=estatistica,
        p_valor=p_valor,
        alpha=alpha,
        interpretacao_rejeita=(
            f"As médias de {value_column} diferem significativamente entre os grupos "
            f"'{group_a}' (média={amostra_a.mean():.2f}) e '{group_b}' "
            f"(média={amostra_b.mean():.2f})."
        ),
        interpretacao_nao_rejeita=(
            f"Não há evidência estatística suficiente de diferença entre as médias de "
            f"{value_column} nos grupos '{group_a}' e '{group_b}'."
        ),
    )


def mann_whitney_test(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    group_a,
    group_b,
    alpha: float = ALPHA_PADRAO,
) -> dict:
    """
    Teste não paramétrico de Mann-Whitney U, alternativa ao t-test quando os
    pressupostos de normalidade/variância não são adequados — compara as
    distribuições (via postos/ranks) de dois grupos independentes.
    """

    dados = df[[value_column, group_column]].dropna()
    amostra_a = pd.to_numeric(dados.loc[dados[group_column] == group_a, value_column], errors="coerce").dropna()
    amostra_b = pd.to_numeric(dados.loc[dados[group_column] == group_b, value_column], errors="coerce").dropna()

    estatistica, p_valor = stats.mannwhitneyu(amostra_a, amostra_b, alternative="two-sided")

    return _linha_resultado(
        nome_teste=f"Mann-Whitney U ({value_column}: {group_a} vs {group_b})",
        h0=f"As distribuições de {value_column} são iguais entre os grupos '{group_a}' e '{group_b}'",
        h1=f"As distribuições de {value_column} diferem entre os grupos '{group_a}' e '{group_b}'",
        estatistica=estatistica,
        p_valor=p_valor,
        alpha=alpha,
        interpretacao_rejeita=(
            f"As distribuições de {value_column} diferem significativamente entre os "
            f"grupos '{group_a}' (mediana={amostra_a.median():.2f}) e '{group_b}' "
            f"(mediana={amostra_b.median():.2f})."
        ),
        interpretacao_nao_rejeita=(
            f"Não há evidência estatística suficiente de diferença entre as "
            f"distribuições de {value_column} nos grupos '{group_a}' e '{group_b}'."
        ),
    )


def anova_test(
    df: pd.DataFrame, value_column: str, group_column: str, alpha: float = ALPHA_PADRAO
) -> dict:
    """
    ANOVA de um fator: testa se a média de ``value_column`` difere entre
    três ou mais grupos definidos por ``group_column``.
    """

    dados = df[[value_column, group_column]].dropna()
    grupos = [
        pd.to_numeric(grupo[value_column], errors="coerce").dropna()
        for _, grupo in dados.groupby(group_column)
    ]
    grupos = [g for g in grupos if len(g) > 1]

    estatistica, p_valor = stats.f_oneway(*grupos)

    return _linha_resultado(
        nome_teste=f"ANOVA one-way ({value_column} by {group_column})",
        h0=f"A média de {value_column} é igual entre todos os grupos de {group_column}",
        h1=f"A média de {value_column} difere entre pelo menos dois grupos de {group_column}",
        estatistica=estatistica,
        p_valor=p_valor,
        alpha=alpha,
        interpretacao_rejeita=(
            f"Há diferença estatisticamente significativa na média de {value_column} "
            f"entre pelo menos dois grupos de {group_column}. Testes post-hoc (ex.: "
            f"Tukey HSD) seriam necessários para identificar quais pares diferem."
        ),
        interpretacao_nao_rejeita=(
            f"Não há evidência estatística suficiente de diferença na média de "
            f"{value_column} entre os grupos de {group_column}."
        ),
    )


def run_all_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a bateria padrão de testes estatísticos do projeto, orientada
    pelas perguntas de negócio da EDA (ver README / notebook 04).
    """

    resultados = []

    # Pearson e Spearman são reportados lado a lado para income/purchase_count
    # x total_spent propositalmente: os outliers extremos injetados nessas
    # variáveis distorcem a correlação linear (Pearson), enquanto a
    # correlação por postos (Spearman) permanece robusta e revela a relação
    # monotônica real — um achado relevante discutido no notebook de EDA.
    resultados.append(pearson_correlation_test(df, "income", "total_spent"))
    resultados.append(spearman_correlation_test(df, "income", "total_spent"))
    resultados.append(pearson_correlation_test(df, "purchase_count", "total_spent"))
    resultados.append(spearman_correlation_test(df, "purchase_count", "total_spent"))
    resultados.append(spearman_correlation_test(df, "satisfaction_score", "total_spent"))
    resultados.append(spearman_correlation_test(df, "age", "total_spent"))

    if "customer_segment" in df.columns:
        segmentos = df["customer_segment"].dropna().unique().tolist()
        if len(segmentos) >= 2:
            resultados.append(
                mann_whitney_test(
                    df, "satisfaction_score", "customer_segment", segmentos[0], segmentos[-1]
                )
            )
        if len(segmentos) >= 3:
            resultados.append(anova_test(df, "income", "customer_segment"))

    if "gender" in df.columns:
        categorias_genero = [g for g in ["masculino", "feminino"] if g in df["gender"].unique()]
        if len(categorias_genero) == 2:
            resultados.append(
                independent_t_test(df, "total_spent", "gender", categorias_genero[0], categorias_genero[1])
            )

    resultado_final = pd.DataFrame(resultados)
    logger.info("Statistical tests executed: %s", len(resultado_final))
    return resultado_final
