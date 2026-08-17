"""
Módulo de deduplicação.

Implementa três níveis de detecção de duplicidade, do mais simples/seguro
ao mais investigativo:

1. Duplicidade completa (todas as colunas idênticas) — removida com
   segurança.
2. Duplicidade por chave de negócio (``customer_id``) — mantém o registro
   mais recente, com base em ``signup_date``.
3. Duplicidade aproximada (fuzzy matching) por nome — apenas reportada para
   investigação manual, nunca removida automaticamente.
"""

from __future__ import annotations

import logging
import unicodedata
from itertools import combinations

import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


def _normalizar_nome(nome: str) -> str:
    """Normaliza um nome para comparação fuzzy: minúsculas e sem acentuação."""

    nfkd = unicodedata.normalize("NFKD", str(nome))
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    return sem_acento.strip().lower()


def find_full_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna as linhas que são duplicidades completas (todas as colunas iguais)."""

    mascara = df.duplicated(keep=False)
    return df.loc[mascara].sort_values(list(df.columns))


def remove_full_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicidades completas, mantendo a primeira ocorrência."""

    linhas_antes = len(df)
    df_dedup = df.drop_duplicates(keep="first").reset_index(drop=True)
    removidos = linhas_antes - len(df_dedup)
    logger.info("Full duplicates removed: %s", removidos)
    return df_dedup, removidos


def find_key_duplicates(df: pd.DataFrame, key: str = "customer_id") -> pd.DataFrame:
    """Retorna todas as linhas cujo valor de ``key`` aparece mais de uma vez."""

    mascara = df.duplicated(subset=[key], keep=False)
    return df.loc[mascara].sort_values(key)


def deduplicate_by_key(
    df: pd.DataFrame,
    key: str = "customer_id",
    sort_column: str = "signup_date",
    keep: str = "last",
) -> tuple[pd.DataFrame, int]:
    """
    Remove duplicidades pela chave de negócio ``key``, mantendo o registro
    mais recente segundo ``sort_column`` (por padrão, a última data de
    cadastro/atualização disponível).
    """

    linhas_antes = len(df)
    df_ordenado = df.sort_values(sort_column)
    df_dedup = df_ordenado.drop_duplicates(subset=[key], keep=keep).reset_index(drop=True)
    removidos = linhas_antes - len(df_dedup)
    logger.info("Key duplicates removed (by %s): %s", key, removidos)
    return df_dedup, removidos


def find_fuzzy_duplicates(
    df: pd.DataFrame,
    column: str = "customer_name",
    threshold: float = 90.0,
    id_column: str = "customer_id",
    max_comparisons_per_block: int = 5000,
) -> pd.DataFrame:
    """
    Identifica possíveis duplicidades aproximadas (mesma pessoa cadastrada
    sob ``customer_id`` diferentes) comparando similaridade textual do nome
    com RapidFuzz. Para manter a comparação tratável em bases maiores, os
    registros são primeiro agrupados por letra inicial do nome (blocking).

    Este relatório é apenas investigativo: nenhuma remoção automática é
    realizada — cabe a um analista validar cada par antes de agir.

    Returns
    -------
    pd.DataFrame
        Colunas: record_1, record_2, name_1, name_2, similarity_score,
        possible_duplicate.
    """

    colunas_saida = ["record_1", "record_2", "name_1", "name_2", "similarity_score", "possible_duplicate"]

    registros = df[[id_column, column]].dropna(subset=[column]).copy()
    if registros.empty:
        return pd.DataFrame(columns=colunas_saida)

    # Normaliza (minúsculas, sem acento) apenas para blocking e cálculo do
    # score — os nomes originais são preservados na saída para investigação.
    registros["_nome_normalizado"] = registros[column].map(_normalizar_nome)
    registros["_bloco"] = registros["_nome_normalizado"].str[:1]

    pares = []
    for _, grupo in registros.groupby("_bloco"):
        indices = grupo.index.tolist()
        if len(indices) > max_comparisons_per_block:
            logger.warning(
                "Block '%s' has %s records; truncating comparisons for performance.",
                grupo["_bloco"].iloc[0],
                len(indices),
            )
            indices = indices[:max_comparisons_per_block]

        for i, j in combinations(indices, 2):
            nome_norm_1 = registros.loc[i, "_nome_normalizado"]
            nome_norm_2 = registros.loc[j, "_nome_normalizado"]
            score = fuzz.token_sort_ratio(nome_norm_1, nome_norm_2)
            if score >= threshold:
                pares.append(
                    {
                        "record_1": registros.loc[i, id_column],
                        "record_2": registros.loc[j, id_column],
                        "name_1": registros.loc[i, column],
                        "name_2": registros.loc[j, column],
                        "similarity_score": round(score, 2),
                        "possible_duplicate": score >= threshold,
                    }
                )

    if not pares:
        return pd.DataFrame(columns=colunas_saida)

    resultado = pd.DataFrame(pares).sort_values("similarity_score", ascending=False, ignore_index=True)
    logger.info("Fuzzy duplicate candidates found: %s", len(resultado))
    return resultado


def deduplication_summary(
    linhas_originais: int,
    removidos_completos: int,
    removidos_por_chave: int,
    candidatos_fuzzy: int,
) -> pd.DataFrame:
    """Consolida um resumo textual do processo de deduplicação."""

    return pd.DataFrame(
        [
            {"etapa": "Registros originais (RAW)", "quantidade": linhas_originais},
            {"etapa": "Duplicidades completas removidas", "quantidade": removidos_completos},
            {"etapa": "Duplicidades por customer_id removidas", "quantidade": removidos_por_chave},
            {
                "etapa": "Registros após deduplicação determinística",
                "quantidade": linhas_originais - removidos_completos - removidos_por_chave,
            },
            {
                "etapa": "Candidatos a duplicidade aproximada (apenas investigação)",
                "quantidade": candidatos_fuzzy,
            },
        ]
    )
