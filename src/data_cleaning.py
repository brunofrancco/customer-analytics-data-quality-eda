"""
Módulo de Data Wrangling / Data Cleaning.

Cada função é independente, recebe um DataFrame, aplica uma única
transformação bem definida e retorna um novo DataFrame — o que facilita
testes unitários e reuso em diferentes contextos (pipeline, notebooks,
scripts ad-hoc).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

IDADE_MINIMA_VALIDA = 18
IDADE_MAXIMA_VALIDA = 100

# ---------------------------------------------------------------------------
# Dicionários de mapeamento para padronização de categorias
# ---------------------------------------------------------------------------

MAPA_GENERO = {
    "m": "masculino",
    "masc": "masculino",
    "masculino": "masculino",
    "male": "masculino",
    "f": "feminino",
    "fem": "feminino",
    "feminino": "feminino",
    "female": "feminino",
    "outro": "outro",
    "nao binario": "outro",
    "não binário": "outro",
    "prefer not to say": "outro",
    "n/i": "outro",
}

# UF -> lista de variantes conhecidas (nome completo, com/sem acento, etc.)
MAPA_ESTADOS_UF = {
    "SP": ["sp", "sao paulo"],
    "RJ": ["rj", "rio de janeiro"],
    "MG": ["mg", "minas gerais"],
    "BA": ["ba", "bahia"],
    "RS": ["rs", "rio grande do sul"],
    "PR": ["pr", "parana"],
    "PE": ["pe", "pernambuco"],
    "CE": ["ce", "ceara"],
    "SC": ["sc", "santa catarina"],
    "GO": ["go", "goias"],
    "DF": ["df", "distrito federal"],
    "ES": ["es", "espirito santo"],
    "PA": ["pa", "para"],
    "AM": ["am", "amazonas"],
    "MT": ["mt", "mato grosso"],
}

# Constrói o mapa reverso (variante normalizada -> UF) uma única vez.
_MAPA_ESTADOS_REVERSO: dict[str, str] = {
    variante: uf for uf, variantes in MAPA_ESTADOS_UF.items() for variante in variantes
}


def _remover_acentos(texto: str) -> str:
    """Remove acentuação de uma string, preservando as demais letras."""

    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalizar_chave(texto: str) -> str:
    """Normaliza uma string para uso como chave de mapeamento (lower + sem acento + sem espaços extras)."""

    return _remover_acentos(str(texto)).strip().lower()


# ---------------------------------------------------------------------------
# 1. Nomes de colunas
# ---------------------------------------------------------------------------

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza os nomes das colunas para snake_case (minúsculas, com
    underscore), removendo espaços extras e caracteres especiais.

    Exemplo: "Customer ID" -> "customer_id"; "Total Spent" -> "total_spent".
    """

    df = df.copy()
    novos_nomes = {}
    for coluna in df.columns:
        nome = _remover_acentos(str(coluna)).strip().lower()
        nome = re.sub(r"[^a-z0-9]+", "_", nome)
        nome = re.sub(r"_+", "_", nome).strip("_")
        novos_nomes[coluna] = nome

    df = df.rename(columns=novos_nomes)
    logger.info("Column names standardized: %s", list(df.columns))
    return df


# ---------------------------------------------------------------------------
# 2. Texto genérico
# ---------------------------------------------------------------------------

def clean_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Remove espaços no início/fim e colapsa espaços duplicados em colunas de
    texto. Não altera capitalização (isso é feito por funções específicas,
    como ``standardize_gender``/``standardize_state``, quando aplicável).
    """

    df = df.copy()
    for coluna in columns:
        if coluna not in df.columns:
            continue
        df[coluna] = (
            df[coluna]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    return df


def standardize_customer_name(df: pd.DataFrame, column: str = "customer_name") -> pd.DataFrame:
    """Padroniza a capitalização dos nomes de clientes (Title Case)."""

    df = df.copy()
    if column in df.columns:
        df[column] = df[column].astype("string").str.title()
    return df


# ---------------------------------------------------------------------------
# 3. Categorias: gênero e estado
# ---------------------------------------------------------------------------

def standardize_gender(df: pd.DataFrame, column: str = "gender") -> pd.DataFrame:
    """
    Padroniza a coluna de gênero para as categorias canônicas:
    'masculino', 'feminino' ou 'outro'. Valores não reconhecidos viram
    'outro' e são registrados em log.
    """

    df = df.copy()
    if column not in df.columns:
        return df

    chaves_normalizadas = df[column].astype("string").map(
        lambda v: _normalizar_chave(v) if pd.notna(v) else v
    )
    categoria_padronizada = chaves_normalizadas.map(MAPA_GENERO)

    nao_reconhecidos = chaves_normalizadas[
        categoria_padronizada.isna() & chaves_normalizadas.notna()
    ].unique()
    if len(nao_reconhecidos) > 0:
        logger.warning("Unrecognized gender values mapped to 'outro': %s", list(nao_reconhecidos))

    categoria_padronizada = categoria_padronizada.fillna(
        chaves_normalizadas.where(chaves_normalizadas.isna(), "outro")
    )
    df[column] = categoria_padronizada
    return df


def standardize_state(df: pd.DataFrame, column: str = "state") -> pd.DataFrame:
    """
    Padroniza a coluna de estado (UF) para a sigla de 2 letras (ex.: "SP"),
    reconhecendo o nome completo do estado, com ou sem acentuação, e
    diferentes capitalizações.
    """

    df = df.copy()
    if column not in df.columns:
        return df

    chaves_normalizadas = df[column].astype("string").map(
        lambda v: _normalizar_chave(v) if pd.notna(v) else v
    )
    uf_padronizada = chaves_normalizadas.map(_MAPA_ESTADOS_REVERSO)

    nao_reconhecidos = chaves_normalizadas[uf_padronizada.isna() & chaves_normalizadas.notna()].unique()
    if len(nao_reconhecidos) > 0:
        logger.warning("Unrecognized state values kept as 'unknown': %s", list(nao_reconhecidos))

    uf_padronizada = uf_padronizada.where(uf_padronizada.notna() | chaves_normalizadas.isna(), "unknown")
    df[column] = uf_padronizada
    return df


# ---------------------------------------------------------------------------
# 4. E-mail
# ---------------------------------------------------------------------------

def clean_email(df: pd.DataFrame, column: str = "email") -> pd.DataFrame:
    """
    Normaliza a coluna de e-mail: remove espaços extras e converte para
    minúsculas. Também sinaliza e-mails com formato inválido (sem "@" ou
    sem domínio) em uma coluna auxiliar ``email_valido``.
    """

    df = df.copy()
    if column not in df.columns:
        return df

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.lower()
    )

    padrao_email = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    df["email_valido"] = df[column].apply(
        lambda v: bool(padrao_email.match(v)) if pd.notna(v) else False
    )
    quantidade_invalida = int((~df["email_valido"]).sum())
    if quantidade_invalida:
        logger.warning("Invalid e-mails found: %s", quantidade_invalida)

    return df


# ---------------------------------------------------------------------------
# 5. Conversão de valores numéricos (incluindo formato monetário)
# ---------------------------------------------------------------------------

def _parse_monetary_value(valor) -> float:
    """
    Converte um valor numérico representado como string em diferentes
    formatos para float:
      - "4500"          -> 4500.0
      - "5.200"         -> 5200.0   (ponto como separador de milhar)
      - "R$ 6.300,00"    -> 6300.0   (formato monetário brasileiro)
    """

    if pd.isna(valor):
        return np.nan

    if isinstance(valor, (int, float)):
        return float(valor)

    texto = str(valor).strip()
    texto = texto.replace("R$", "").strip()
    texto = texto.replace(" ", "")

    if not texto:
        return np.nan

    if "," in texto:
        # Formato brasileiro: ponto = milhar, vírgula = decimal.
        texto = texto.replace(".", "").replace(",", ".")
    else:
        # Sem vírgula: se houver ponto, tratamos como separador de milhar
        # quando seguido de exatamente 3 dígitos (ex.: "5.200"); caso
        # contrário, mantemos como separador decimal (ex.: "5.2").
        partes = texto.split(".")
        if len(partes) > 1 and len(partes[-1]) == 3:
            texto = texto.replace(".", "")

    try:
        return float(texto)
    except ValueError:
        return np.nan


def convert_numeric_columns(
    df: pd.DataFrame,
    monetary_columns: Iterable[str] = ("income", "total_spent"),
    plain_numeric_columns: Iterable[str] = ("purchase_count", "satisfaction_score", "age"),
) -> pd.DataFrame:
    """
    Converte colunas numéricas armazenadas como texto para tipos numéricos
    apropriados. Colunas monetárias passam pelo parser de valores em Real
    (R$); as demais são convertidas diretamente com ``pd.to_numeric``.
    """

    df = df.copy()

    for coluna in monetary_columns:
        if coluna in df.columns:
            df[coluna] = df[coluna].apply(_parse_monetary_value)

    for coluna in plain_numeric_columns:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    if "purchase_count" in df.columns:
        df["purchase_count"] = df["purchase_count"].round().astype("Int64")

    return df


# ---------------------------------------------------------------------------
# 6. Datas
# ---------------------------------------------------------------------------

_FORMATOS_DATA_CONHECIDOS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]


def _parse_data_flexivel(valor) -> pd.Timestamp:
    if pd.isna(valor):
        return pd.NaT
    texto = str(valor).strip()
    for formato in _FORMATOS_DATA_CONHECIDOS:
        try:
            return pd.to_datetime(texto, format=formato)
        except (ValueError, TypeError):
            continue
    # Última tentativa: deixa o pandas inferir (mais lento, usado como fallback).
    return pd.to_datetime(texto, errors="coerce")


def convert_dates(df: pd.DataFrame, column: str = "signup_date") -> pd.DataFrame:
    """
    Converte a coluna de data para ``datetime64`` reconhecendo múltiplos
    formatos de entrada, e cria colunas derivadas úteis para análise
    temporal: ano, mês, trimestre e dia da semana.
    """

    df = df.copy()
    if column not in df.columns:
        return df

    df[column] = df[column].apply(_parse_data_flexivel)

    quantidade_invalida = int(df[column].isna().sum())
    if quantidade_invalida:
        logger.warning("Dates that could not be parsed: %s", quantidade_invalida)

    df[f"{column}_year"] = df[column].dt.year
    df[f"{column}_month"] = df[column].dt.month
    df[f"{column}_quarter"] = df[column].dt.quarter
    df[f"{column}_weekday"] = df[column].dt.day_name()

    return df


# ---------------------------------------------------------------------------
# 7. Validação de idade
# ---------------------------------------------------------------------------

def validate_age(
    df: pd.DataFrame,
    column: str = "age",
    minimo: int = IDADE_MINIMA_VALIDA,
    maximo: int = IDADE_MAXIMA_VALIDA,
) -> tuple[pd.DataFrame, dict]:
    """
    Valida a coluna de idade segundo a regra de negócio
    ``18 <= age <= 100``. Idades fora do intervalo NÃO são removidas
    silenciosamente: são convertidas para ausente (NaN) para posterior
    imputação em ``handle_missing_values``, e a ocorrência é registrada.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame atualizado e um relatório com quantidade/percentual de
        idades inválidas e a estratégia aplicada.
    """

    df = df.copy()
    if column not in df.columns:
        return df, {}

    total = len(df)
    mascara_invalida = df[column].notna() & ~df[column].between(minimo, maximo)
    quantidade_invalida = int(mascara_invalida.sum())

    df.loc[mascara_invalida, column] = np.nan

    relatorio = {
        "coluna": column,
        "regra": f"{minimo} <= age <= {maximo}",
        "quantidade_invalida": quantidade_invalida,
        "percentual_invalido": round(100 * quantidade_invalida / total, 2) if total else 0.0,
        "estrategia": "Convertido para ausente (NaN) e imputado pela mediana em handle_missing_values",
    }
    logger.info(
        "Age validation: %s invalid values (%.2f%%)",
        quantidade_invalida,
        relatorio["percentual_invalido"],
    )
    return df, relatorio


# ---------------------------------------------------------------------------
# 8. Tratamento de valores ausentes
# ---------------------------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    group_column_for_income: str = "state",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trata valores ausentes de forma diferenciada por coluna, evitando o uso
    ingênuo de ``fillna(0)``:

    - ``age``: imputação pela mediana global (variável aproximadamente
      simétrica; a mediana é robusta a eventuais outliers residuais).
    - ``income``: imputação pela mediana por grupo (``state``), com fallback
      para a mediana global quando o grupo não possui dados suficientes —
      a renda varia sistematicamente entre estados, então a imputação por
      grupo preserva melhor essa estrutura do que uma mediana única.
    - ``satisfaction_score``: imputação pela mediana global.
    - ``city``: preenchida com a categoria ``"unknown"``, pois não é
      seguro inferir a cidade de um cliente a partir de outras colunas.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        DataFrame tratado e uma tabela (coluna, missing_%, estratégia)
        documentando a decisão tomada para cada coluna.
    """

    df = df.copy()
    total = len(df)
    linhas_relatorio = []

    def _registrar(coluna: str, percentual_antes: float, estrategia: str) -> None:
        linhas_relatorio.append(
            {"coluna": coluna, "missing_percent": percentual_antes, "estrategia": estrategia}
        )

    if "age" in df.columns:
        percentual = round(100 * df["age"].isna().sum() / total, 2) if total else 0.0
        mediana = df["age"].median()
        df["age"] = df["age"].fillna(mediana)
        _registrar("age", percentual, f"mediana global ({mediana:.1f})")

    if "income" in df.columns:
        percentual = round(100 * df["income"].isna().sum() / total, 2) if total else 0.0
        mediana_global = df["income"].median()
        if group_column_for_income in df.columns:
            medianas_grupo = df.groupby(group_column_for_income)["income"].transform("median")
            df["income"] = df["income"].fillna(medianas_grupo)
            estrategia = f"mediana por grupo ({group_column_for_income}), fallback mediana global"
        else:
            estrategia = "mediana global"
        df["income"] = df["income"].fillna(mediana_global)
        _registrar("income", percentual, estrategia)

    if "satisfaction_score" in df.columns:
        percentual = round(100 * df["satisfaction_score"].isna().sum() / total, 2) if total else 0.0
        mediana = df["satisfaction_score"].median()
        df["satisfaction_score"] = df["satisfaction_score"].fillna(mediana)
        _registrar("satisfaction_score", percentual, f"mediana global ({mediana:.1f})")

    if "city" in df.columns:
        percentual = round(100 * df["city"].isna().sum() / total, 2) if total else 0.0
        df["city"] = df["city"].astype("string").fillna("unknown")
        _registrar("city", percentual, "categoria 'unknown'")

    relatorio = pd.DataFrame(linhas_relatorio)
    logger.info("Missing values treated for columns: %s", relatorio["coluna"].tolist() if not relatorio.empty else [])
    return df, relatorio
