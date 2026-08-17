"""Testes para o módulo src.data_cleaning."""

import numpy as np
import pandas as pd

from src.data_cleaning import (
    clean_email,
    clean_text_columns,
    convert_dates,
    convert_numeric_columns,
    handle_missing_values,
    standardize_column_names,
    standardize_customer_name,
    standardize_gender,
    standardize_state,
    validate_age,
)


def test_standardize_column_names(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    esperado = [
        "customer_id", "customer_name", "email", "age", "gender", "city",
        "state", "signup_date", "income", "purchase_count", "total_spent",
        "satisfaction_score",
    ]
    assert list(df.columns) == esperado


def test_clean_text_columns_remove_espacos(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    df = clean_text_columns(df, ["customer_name"])
    # "  Pedro   Alves  " -> "Pedro Alves"
    assert df.loc[3, "customer_name"] == "Pedro Alves"


def test_standardize_customer_name_title_case(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    df = clean_text_columns(df, ["customer_name"])
    df = standardize_customer_name(df)
    assert df.loc[0, "customer_name"] == "João Silva"
    assert df.loc[1, "customer_name"] == "Joao Silva"


def test_standardize_gender_mapeia_variantes(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    df = standardize_gender(df)
    assert set(df["gender"].unique()).issubset({"masculino", "feminino", "outro"})
    assert df.loc[0, "gender"] == "masculino"  # "M"
    assert df.loc[1, "gender"] == "masculino"  # "male"
    assert df.loc[2, "gender"] == "feminino"   # "F"
    assert df.loc[3, "gender"] == "feminino"   # "Fem"


def test_standardize_state_mapeia_variantes(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    df = standardize_state(df)
    assert set(df["state"].unique()) == {"SP", "RJ", "PR"}


def test_clean_email_normaliza_e_valida(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    df = clean_email(df)
    # As três variações "JOAO@EMAIL.COM" / " joao@email.com" / "JOAO@EMAIL.COM"
    # (linhas 0, 1 e 4) devem convergir para o mesmo e-mail normalizado.
    assert (df["email"] == "joao@email.com").sum() == 3
    assert df["email_valido"].all()


def test_convert_numeric_columns_formatos_monetarios(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    df = convert_numeric_columns(df)
    # "4500", "R$ 4.500,00" e "4500" devem virar todos 4500.0
    assert df.loc[0, "income"] == 4500.0
    assert df.loc[1, "income"] == 4500.0
    # "5.200" (separador de milhar) deve virar 5200.0
    assert df.loc[2, "income"] == 5200.0
    # "R$ 5.000,00" deve virar 5000.0
    assert df.loc[3, "total_spent"] == 5000.0
    assert df["income"].dtype == np.float64


def test_convert_dates_multiplos_formatos(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    df = convert_dates(df)
    assert pd.api.types.is_datetime64_any_dtype(df["signup_date"])
    # Todas as datas de exemplo referem-se a janeiro/fevereiro/março de 2024.
    assert df["signup_date"].notna().all()
    assert set(df["signup_date_year"].unique()) == {2024}


def test_validate_age_marca_invalidos_sem_remover_linhas(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    linhas_antes = len(df)
    df, relatorio = validate_age(df)

    assert len(df) == linhas_antes  # nenhuma linha removida
    assert pd.isna(df.loc[2, "age"])  # -5 vira NaN
    assert pd.isna(df.loc[3, "age"])  # 150 vira NaN
    assert relatorio["quantidade_invalida"] == 2


def test_handle_missing_values_nao_usa_fillna_zero(df_bruto_exemplo):
    df = standardize_column_names(df_bruto_exemplo)
    df = convert_numeric_columns(df)
    df, relatorio = validate_age(df)
    df, relatorio_missing = handle_missing_values(df)

    assert df["age"].isna().sum() == 0
    assert df["city"].isna().sum() == 0
    assert (df["city"] == "unknown").sum() == 1
    # A idade imputada não deve ser 0 (o que indicaria um fillna(0) ingênuo).
    assert df["age"].min() > 0
    assert not relatorio_missing.empty
