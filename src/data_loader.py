"""
Módulo responsável pelo carregamento (ingestão) dos dados brutos e
processados do projeto.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DIR_RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_RAW = DIR_RAIZ / "data" / "raw" / "customers_raw.csv"
CAMINHO_PROCESSED = DIR_RAIZ / "data" / "processed" / "customers_clean.csv"
DIR_OUTPUT = DIR_RAIZ / "data" / "output"
DIR_FIGURES = DIR_RAIZ / "reports" / "figures"


def load_raw_data(caminho: Path | str = CAMINHO_RAW) -> pd.DataFrame:
    """
    Carrega o dataset bruto (RAW) de clientes.

    Parameters
    ----------
    caminho:
        Caminho para o arquivo CSV bruto. Por padrão, usa
        ``data/raw/customers_raw.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame com os dados exatamente como recebidos (sem qualquer
        tratamento), preservando colunas como texto sempre que houver
        ambiguidade de tipo.
    """

    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de dados brutos não encontrado em '{caminho}'. "
            "Execute 'python -m src.generate_synthetic_data' primeiro."
        )

    logger.info("Loading dataset from %s", caminho)
    df = pd.read_csv(caminho, dtype=str, keep_default_na=True)
    logger.info("Dataset loaded: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def load_processed_data(caminho: Path | str = CAMINHO_PROCESSED) -> pd.DataFrame:
    """Carrega o dataset já tratado (processed), se existir."""

    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo processado não encontrado em '{caminho}'. "
            "Execute 'python run_pipeline.py' primeiro."
        )

    df = pd.read_csv(caminho, parse_dates=["signup_date"])
    return df


def save_dataframe(df: pd.DataFrame, caminho: Path | str, index: bool = False) -> None:
    """Salva um DataFrame em CSV, criando as pastas necessárias."""

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=index, encoding="utf-8")
    logger.info("Saved file: %s (%s rows)", caminho, len(df))
