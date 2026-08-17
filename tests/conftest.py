"""Fixtures compartilhadas entre os testes do projeto."""

import pandas as pd
import pytest


@pytest.fixture
def df_bruto_exemplo() -> pd.DataFrame:
    """
    Pequena amostra de dados 'sujos', representativa dos problemas descritos
    no briefing do projeto (usada para testar as funções de limpeza de
    forma isolada e determinística).
    """

    return pd.DataFrame(
        {
            "Customer ID": ["CUST000001", "CUST000002", "CUST000003", "CUST000004", "CUST000001"],
            "Customer Name": ["JOÃO SILVA", "joao silva", "Maria Souza", "  Pedro   Alves  ", "JOÃO SILVA"],
            "Email": ["JOAO@EMAIL.COM", " joao@email.com", "maria@EMAIL.com", "pedro@email.com", "JOAO@EMAIL.COM"],
            "Age": [30, 30, -5, 150, 30],
            "Gender": ["M", "male", "F", "Fem", "M"],
            "City": ["São Paulo", None, "Rio de Janeiro", "Curitiba", "São Paulo"],
            "State": ["SP", "sp", "Rio de Janeiro", "pr", "SP"],
            "Signup Date": ["2024-01-10", "10/01/2024", "2024/02/15", "15-03-2024", "2024-01-10"],
            "Income": ["4500", "R$ 4.500,00", "5.200", "R$ 3.100,00", "4500"],
            "Purchase Count": [5, 5, 3, 10, 5],
            "Total Spent": ["1200", "1200", "900", "R$ 5.000,00", "1200"],
            "Satisfaction Score": [4.5, 4.5, 3.0, None, 4.5],
        }
    )


@pytest.fixture
def df_limpo_exemplo() -> pd.DataFrame:
    """DataFrame já no formato 'processado' (colunas padronizadas), para testes de data quality."""

    return pd.DataFrame(
        {
            "customer_id": ["CUST000001", "CUST000002", "CUST000003", "CUST000004"],
            "customer_name": ["João Silva", "Maria Souza", "Pedro Alves", "Ana Lima"],
            "email": ["joao@email.com", "maria@email.com", "pedro@email.com", "ana@email.com"],
            "email_valido": [True, True, True, True],
            "age": [30, 45, 29, 60],
            "gender": ["masculino", "feminino", "masculino", "feminino"],
            "city": ["São Paulo", "Rio de Janeiro", "Curitiba", "unknown"],
            "state": ["SP", "RJ", "PR", "SP"],
            "signup_date": pd.to_datetime(["2024-01-10", "2024-02-15", "2024-03-15", "2024-04-01"]),
            "income": [4500.0, 5200.0, 3100.0, 6000.0],
            "purchase_count": [5, 3, 10, 2],
            "total_spent": [1200.0, 900.0, 5000.0, 300.0],
            "satisfaction_score": [4.5, 3.0, 4.0, 5.0],
        }
    )
