"""Testes para o módulo src.data_quality."""

import numpy as np
import pandas as pd

from src.data_quality import (
    check_completeness,
    check_consistency,
    check_uniqueness,
    check_validity,
    data_quality_score,
    generate_data_quality_report,
)


def test_check_completeness_percentuais(df_limpo_exemplo):
    df = df_limpo_exemplo.copy()
    df.loc[0, "city"] = np.nan
    relatorio = check_completeness(df)

    linha_city = relatorio.loc[relatorio["metric"] == "completeness_city"].iloc[0]
    assert linha_city["value"] == 75.0  # 3 de 4 preenchidos


def test_check_uniqueness_sem_duplicidade(df_limpo_exemplo):
    relatorio = check_uniqueness(df_limpo_exemplo, key="customer_id")
    assert relatorio.iloc[0]["value"] == 100.0


def test_check_uniqueness_com_duplicidade(df_limpo_exemplo):
    df = pd.concat([df_limpo_exemplo, df_limpo_exemplo.iloc[[0]]], ignore_index=True)
    relatorio = check_uniqueness(df, key="customer_id")
    assert relatorio.iloc[0]["value"] < 100.0


def test_check_validity_idade_dentro_da_regra(df_limpo_exemplo):
    relatorio = check_validity(df_limpo_exemplo)
    linha_idade = relatorio.loc[relatorio["metric"] == "validity_age"].iloc[0]
    assert linha_idade["value"] == 100.0  # todas as idades do fixture são válidas (18-100)


def test_check_validity_detecta_idade_invalida(df_limpo_exemplo):
    df = df_limpo_exemplo.copy()
    df.loc[0, "age"] = 150  # idade inválida, deveria já ter sido tratada antes desta etapa
    relatorio = check_validity(df)
    linha_idade = relatorio.loc[relatorio["metric"] == "validity_age"].iloc[0]
    assert linha_idade["value"] < 100.0


def test_check_consistency_categorias_de_genero(df_limpo_exemplo):
    df = df_limpo_exemplo.copy()
    df.loc[0, "gender"] = "M"  # categoria não padronizada
    relatorio = check_consistency(df)
    linha_genero = relatorio.loc[relatorio["metric"] == "consistency_gender_categories"].iloc[0]
    assert linha_genero["value"] == 75.0  # 3 de 4 estão na categoria padronizada


def test_generate_data_quality_report_sem_valores_impossiveis(df_limpo_exemplo):
    relatorio = generate_data_quality_report(df_limpo_exemplo)
    assert not relatorio.empty
    assert {"metric", "value", "status", "description"}.issubset(relatorio.columns)

    score = data_quality_score(relatorio)
    assert 0 <= score <= 100
    # Como o fixture já está limpo e consistente, o score deve ser alto.
    assert score >= 90
