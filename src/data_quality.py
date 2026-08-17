"""
Módulo de Data Quality.

Implementa verificações inspiradas nas dimensões clássicas de qualidade de
dados (DAMA-DMBOK): Completeness, Uniqueness, Validity, Consistency e
Accuracy, consolidando tudo em um relatório único
(``data_quality_report.csv``) e um Data Quality Score geral.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

STATUS_OK = "OK"
STATUS_ATENCAO = "ATENÇÃO"
STATUS_CRITICO = "CRÍTICO"


def _classificar_status(valor_percentual: float, limite_ok: float, limite_atencao: float) -> str:
    """
    Classifica um percentual "quanto maior, pior" (ex.: % de erro) em
    OK / ATENÇÃO / CRÍTICO segundo limites definidos.
    """

    if valor_percentual <= limite_ok:
        return STATUS_OK
    if valor_percentual <= limite_atencao:
        return STATUS_ATENCAO
    return STATUS_CRITICO


def check_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Completeness: percentual de valores efetivamente preenchidos por coluna.
    """

    linhas = []
    total = len(df)
    for coluna in df.columns:
        preenchidos = int(df[coluna].notna().sum())
        percentual = round(100 * preenchidos / total, 2) if total else 0.0
        linhas.append(
            {
                "metric": f"completeness_{coluna}",
                "value": percentual,
                "status": _classificar_status(100 - percentual, 2, 10),
                "description": f"Percentual de valores preenchidos na coluna '{coluna}'.",
            }
        )
    return pd.DataFrame(linhas)


def check_uniqueness(df: pd.DataFrame, key: str = "customer_id") -> pd.DataFrame:
    """Uniqueness: percentual de registros únicos considerando a chave de negócio."""

    total = len(df)
    unicos = int(df[key].nunique())
    percentual_unico = round(100 * unicos / total, 2) if total else 0.0
    duplicados = total - unicos

    return pd.DataFrame(
        [
            {
                "metric": "uniqueness_customer_id",
                "value": percentual_unico,
                "status": _classificar_status(100 - percentual_unico, 0, 1),
                "description": f"Percentual de '{key}' únicos ({unicos} de {total} registros; {duplicados} duplicados).",
            }
        ]
    )


def check_validity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validity: percentual de valores dentro das regras de negócio esperadas
    (idade entre 18-100, e-mail em formato válido, satisfação entre 1-5).
    """

    linhas = []
    total = len(df)

    if "age" in df.columns:
        validos = int(df["age"].between(18, 100).sum())
        percentual = round(100 * validos / total, 2) if total else 0.0
        linhas.append(
            {
                "metric": "validity_age",
                "value": percentual,
                "status": _classificar_status(100 - percentual, 1, 5),
                "description": "Percentual de idades dentro do intervalo válido (18-100).",
            }
        )

    if "email_valido" in df.columns:
        percentual = round(100 * df["email_valido"].mean(), 2)
        linhas.append(
            {
                "metric": "validity_email",
                "value": percentual,
                "status": _classificar_status(100 - percentual, 1, 5),
                "description": "Percentual de e-mails em formato válido (usuario@dominio.com).",
            }
        )

    if "satisfaction_score" in df.columns:
        validos = int(df["satisfaction_score"].between(1, 5).sum())
        percentual = round(100 * validos / total, 2) if total else 0.0
        linhas.append(
            {
                "metric": "validity_satisfaction_score",
                "value": percentual,
                "status": _classificar_status(100 - percentual, 1, 5),
                "description": "Percentual de notas de satisfação dentro do intervalo esperado (1-5).",
            }
        )

    if "income" in df.columns:
        validos = int((df["income"] > 0).sum())
        percentual = round(100 * validos / total, 2) if total else 0.0
        linhas.append(
            {
                "metric": "validity_income_positive",
                "value": percentual,
                "status": _classificar_status(100 - percentual, 1, 5),
                "description": "Percentual de rendas com valor positivo (> 0).",
            }
        )

    return pd.DataFrame(linhas)


def check_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consistency: verifica se colunas categóricas foram padronizadas para o
    conjunto esperado de valores, e se relações lógicas entre variáveis se
    mantêm (ex.: clientes com 0 compras mas gasto total > 0).
    """

    linhas = []
    total = len(df)

    if "gender" in df.columns:
        categorias_esperadas = {"masculino", "feminino", "outro"}
        consistentes = int(df["gender"].isin(categorias_esperadas).sum())
        percentual = round(100 * consistentes / total, 2) if total else 0.0
        linhas.append(
            {
                "metric": "consistency_gender_categories",
                "value": percentual,
                "status": _classificar_status(100 - percentual, 0, 2),
                "description": "Percentual de registros com categoria de gênero padronizada.",
            }
        )

    if "state" in df.columns:
        estados_validos = df["state"].str.len().eq(2).fillna(False)
        percentual = round(100 * estados_validos.sum() / total, 2) if total else 0.0
        linhas.append(
            {
                "metric": "consistency_state_format",
                "value": percentual,
                "status": _classificar_status(100 - percentual, 0, 2),
                "description": "Percentual de registros com UF padronizada em 2 letras.",
            }
        )

    if {"purchase_count", "total_spent"}.issubset(df.columns):
        inconsistentes = int(((df["purchase_count"] == 0) & (df["total_spent"] > 0)).sum())
        percentual_inconsistente = round(100 * inconsistentes / total, 2) if total else 0.0
        linhas.append(
            {
                "metric": "consistency_purchase_vs_spent",
                "value": round(100 - percentual_inconsistente, 2),
                "status": _classificar_status(percentual_inconsistente, 0.5, 2),
                "description": (
                    "Percentual de registros consistentes entre 'purchase_count' e "
                    "'total_spent' (não deveria haver gasto sem nenhuma compra registrada)."
                ),
            }
        )

    return pd.DataFrame(linhas)


def check_accuracy_note() -> pd.DataFrame:
    """
    Accuracy: mede o quão corretos os valores são frente à realidade externa
    (ex.: comparar o e-mail cadastrado com o e-mail real do cliente). Como
    este projeto utiliza dados sintéticos e não há uma fonte externa de
    verdade (golden record) disponível para comparação, essa dimensão não
    pode ser medida diretamente — apenas documentada como limitação.
    """

    return pd.DataFrame(
        [
            {
                "metric": "accuracy_note",
                "value": None,
                "status": "N/A",
                "description": (
                    "Accuracy não pôde ser medida diretamente: não há uma fonte de dados "
                    "externa e confiável (golden record) para comparação. Em um cenário "
                    "real, seria necessário validar contra um sistema de referência "
                    "(ex.: Receita Federal para CPF/e-mail, correios para endereço)."
                ),
            }
        ]
    )


def generate_data_quality_report(df: pd.DataFrame, key: str = "customer_id") -> pd.DataFrame:
    """Consolida todas as verificações de qualidade em um único relatório."""

    partes = [
        check_completeness(df),
        check_uniqueness(df, key=key),
        check_validity(df),
        check_consistency(df),
        check_accuracy_note(),
    ]
    relatorio = pd.concat(partes, ignore_index=True)
    logger.info("Data quality report generated with %s metrics", len(relatorio))
    return relatorio


def data_quality_score(relatorio: pd.DataFrame) -> float:
    """
    Calcula um Data Quality Score geral (0-100), como a média das métricas
    percentuais mensuráveis (ignora a métrica de accuracy, que é apenas
    documental/N-A).
    """

    mensuraveis = relatorio[relatorio["value"].notna()]
    if mensuraveis.empty:
        return 0.0
    score = round(float(mensuraveis["value"].mean()), 2)
    logger.info("Overall Data Quality Score: %s", score)
    return score
