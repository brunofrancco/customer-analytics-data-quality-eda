"""Testes para o módulo src.deduplication."""

import pandas as pd

from src.data_cleaning import standardize_column_names
from src.deduplication import (
    deduplicate_by_key,
    deduplication_summary,
    find_fuzzy_duplicates,
    find_full_duplicates,
    find_key_duplicates,
    remove_full_duplicates,
)


def test_find_and_remove_full_duplicates(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    duplicadas = find_full_duplicates(df)
    assert len(duplicadas) == 2  # linhas 0 e 4 são idênticas

    df_dedup, removidos = remove_full_duplicates(df)
    assert removidos == 1
    assert len(df_dedup) == len(df) - 1
    assert df_dedup.duplicated().sum() == 0


def test_find_key_duplicates(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    df, _ = remove_full_duplicates(df)
    # Após remover a duplicidade completa, ainda não há duplicidade de ID
    # neste fixture específico (a única duplicidade era completa).
    duplicadas = find_key_duplicates(df, key="customer_id")
    assert duplicadas.empty


def test_deduplicate_by_key_mantem_registro_mais_recente():
    df = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B"],
            "signup_date": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-02-01"]),
            "purchase_count": [1, 5, 2],
        }
    )
    df_dedup, removidos = deduplicate_by_key(df, key="customer_id", sort_column="signup_date", keep="last")

    assert removidos == 1
    assert df_dedup["customer_id"].nunique() == len(df_dedup)
    registro_a = df_dedup.loc[df_dedup["customer_id"] == "A"].iloc[0]
    assert registro_a["purchase_count"] == 5  # o registro mais recente (2024-06-01)


def test_find_fuzzy_duplicates_encontra_nomes_parecidos():
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "customer_name": ["João Silva", "joao silva", "JOÃO SILVA", "Maria Souza"],
        }
    )
    resultado = find_fuzzy_duplicates(df, column="customer_name", threshold=85.0)

    assert not resultado.empty
    # Todos os pares encontrados devem envolver apenas os registros de "João Silva".
    ids_encontrados = set(resultado["record_1"]).union(resultado["record_2"])
    assert ids_encontrados == {"C1", "C2", "C3"}
    assert "C4" not in ids_encontrados


def test_deduplication_summary_formato():
    resumo = deduplication_summary(100, 10, 5, 3)
    assert list(resumo.columns) == ["etapa", "quantidade"]
    assert len(resumo) == 5
