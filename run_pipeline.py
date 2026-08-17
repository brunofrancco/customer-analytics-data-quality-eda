"""
Pipeline principal do projeto Customer Analytics — Data Quality & EDA.

Executa o fluxo completo de ponta a ponta:

    1. Load (ingestão dos dados brutos)
    2. Profiling (diagnóstico inicial da base RAW)
    3. Cleaning (Data Wrangling)
    4. Deduplication
    5. Data Quality (relatório e score)
    6. Descriptive Statistics
    7. Outlier Detection
    8. Statistical Tests
    9. Export (dados processados, relatórios em data/output/ e figuras em
       reports/figures/)

Uso:
    python run_pipeline.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src import data_loader
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
from src.data_quality import data_quality_score, generate_data_quality_report
from src.deduplication import (
    deduplicate_by_key,
    deduplication_summary,
    find_fuzzy_duplicates,
    remove_full_duplicates,
)
from src.descriptive_stats import (
    COLUNAS_NUMERICAS_PADRAO,
    compute_correlation_matrix,
    compute_descriptive_statistics,
    create_customer_segments,
    detect_outliers_iqr,
    top_correlated_pairs,
)
from src.statistical_tests import run_all_statistical_tests
from src.visualization import (
    plot_bivariate_relationship,
    plot_boxplot,
    plot_correlation_heatmap,
    plot_customer_segmentation,
    plot_customers_by_state,
    plot_distribution,
    plot_missing_values,
    plot_monthly_customers,
    plot_outliers_overview,
)

DIR_RAIZ = Path(__file__).resolve().parent
DIR_OUTPUT = DIR_RAIZ / "data" / "output"
DIR_PROCESSED = DIR_RAIZ / "data" / "processed"
DIR_FIGURES = DIR_RAIZ / "reports" / "figures"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")


def etapa_profiling(df) -> None:
    """Gera um diagnóstico rápido da base RAW (impresso no log)."""

    logger.info("Profiling raw dataset: shape=%s", df.shape)
    nulos = df.isna().sum()
    colunas_com_nulos = nulos[nulos > 0]
    if not colunas_com_nulos.empty:
        logger.info("Columns with missing values (raw): %s", colunas_com_nulos.to_dict())
    duplicadas_completas = int(df.duplicated().sum())
    logger.info("Fully duplicated rows (raw): %s", duplicadas_completas)


def etapa_cleaning(df):
    """Aplica toda a etapa de Data Wrangling / Cleaning."""

    df = standardize_column_names(df)
    df = clean_text_columns(df, ["customer_name", "email", "gender", "city", "state"])
    df = standardize_customer_name(df)
    df = standardize_gender(df)
    df = standardize_state(df)
    df = clean_email(df)
    df = convert_numeric_columns(df)
    df = convert_dates(df)
    df, relatorio_idade = validate_age(df)
    logger.info("Missing values treated")
    return df, relatorio_idade


def etapa_deduplication(df):
    """Aplica deduplicação completa, por chave e investigação de duplicidade aproximada."""

    linhas_originais = len(df)

    df, removidos_completos = remove_full_duplicates(df)
    df, removidos_chave = deduplicate_by_key(df, key="customer_id", sort_column="signup_date")

    total_removidos = removidos_completos + removidos_chave
    logger.info("Duplicates removed: %s", total_removidos)

    candidatos_fuzzy = find_fuzzy_duplicates(df, column="customer_name", threshold=90.0)
    resumo = deduplication_summary(linhas_originais, removidos_completos, removidos_chave, len(candidatos_fuzzy))

    return df, candidatos_fuzzy, resumo


def etapa_visualizacoes(df, corr_pearson) -> None:
    """Gera e salva todas as figuras exigidas pelo projeto."""

    DIR_FIGURES.mkdir(parents=True, exist_ok=True)

    plot_distribution(
        df, "age", "Distribuição de Idade dos Clientes", "Idade (anos)",
        caminho=DIR_FIGURES / "age_distribution.png",
    )
    plot_distribution(
        df, "income", "Distribuição de Renda dos Clientes", "Renda mensal (R$)",
        caminho=DIR_FIGURES / "income_distribution.png",
    )
    plot_distribution(
        df, "total_spent", "Distribuição de Gasto Total dos Clientes", "Gasto total (R$)",
        caminho=DIR_FIGURES / "total_spent_distribution.png",
    )
    plot_boxplot(
        df, "income", "Boxplot de Renda", "Renda mensal (R$)",
        caminho=DIR_FIGURES / "boxplot_income.png", escala_log=True,
    )
    plot_boxplot(
        df, "total_spent", "Boxplot de Gasto Total", "Gasto total (R$)",
        caminho=DIR_FIGURES / "boxplot_total_spent.png", escala_log=True,
    )
    plot_correlation_heatmap(corr_pearson, caminho=DIR_FIGURES / "correlation_heatmap.png")
    plot_customer_segmentation(df, caminho=DIR_FIGURES / "customer_segmentation.png")
    plot_customers_by_state(df, caminho=DIR_FIGURES / "customers_by_state.png")
    plot_monthly_customers(df, caminho=DIR_FIGURES / "monthly_customers.png")
    plot_outliers_overview(
        df, ["income", "total_spent", "purchase_count"], caminho=DIR_FIGURES / "outliers.png"
    )
    plot_bivariate_relationship(
        df, "income", "total_spent",
        "Renda x Gasto Total (por Segmento)", "Renda mensal (R$)", "Gasto total (R$)",
        hue_column="customer_segment", caminho=DIR_FIGURES / "income_vs_total_spent.png",
        escala_log_x=True, escala_log_y=True,
    )
    plot_bivariate_relationship(
        df, "purchase_count", "total_spent",
        "Nº de Compras x Gasto Total", "Número de compras", "Gasto total (R$)",
        caminho=DIR_FIGURES / "purchase_count_vs_total_spent.png",
        escala_log_y=True,
    )
    plot_bivariate_relationship(
        df, "satisfaction_score", "total_spent",
        "Satisfação x Gasto Total", "Nota de satisfação (1-5)", "Gasto total (R$)",
        caminho=DIR_FIGURES / "satisfaction_vs_total_spent.png",
    )
    logger.info("All figures saved to %s", DIR_FIGURES)


def main() -> None:
    inicio = time.time()
    logger.info("=" * 70)
    logger.info("Starting Customer Analytics — Data Quality & EDA pipeline")
    logger.info("=" * 70)

    # 1. LOAD -----------------------------------------------------------
    logger.info("Loading dataset")
    df_raw = data_loader.load_raw_data()
    logger.info("Dataset loaded: %s rows", len(df_raw))

    # 2. PROFILING --------------------------------------------------------
    logger.info("Profiling raw dataset")
    df_raw_padronizado = standardize_column_names(df_raw)
    figura_ausentes_raw = plot_missing_values(df_raw_padronizado, caminho=DIR_FIGURES / "missing_values_raw.png")
    etapa_profiling(df_raw_padronizado)

    # Snapshot de Data Quality ANTES da limpeza, apenas para fins de
    # comparação "antes x depois" (não altera o fluxo de cleaning real).
    # Converte colunas numéricas para permitir os checks de 'validity',
    # mas mantém gender/state/email como no dado bruto, para que os checks
    # de 'consistency' reflitam a inconsistência real de categorias.
    df_raw_para_dq = convert_numeric_columns(df_raw_padronizado.copy())
    relatorio_qualidade_raw = generate_data_quality_report(df_raw_para_dq)
    score_qualidade_raw = data_quality_score(relatorio_qualidade_raw)
    logger.info("Data Quality Score (RAW, before cleaning): %.2f/100", score_qualidade_raw)

    # 3. CLEANING ---------------------------------------------------------
    logger.info("Cleaning data")
    try:
        df_limpo, relatorio_idade = etapa_cleaning(df_raw)
    except Exception:
        logger.exception("Unrecoverable error while cleaning the dataset.")
        raise

    # 4. DEDUPLICATION ------------------------------------------------------
    logger.info("Removing duplicates")
    df_dedup, candidatos_fuzzy, resumo_dedup = etapa_deduplication(df_limpo)
    logger.info("Duplicates removed: %s", resumo_dedup.iloc[[1, 2]]["quantidade"].sum())

    # Trata os valores ausentes (inclui os gerados pela validação de idade)
    # após a deduplicação, para que a imputação reflita a base final.
    df_processado, relatorio_missing = handle_missing_values(df_dedup)
    df_processado = create_customer_segments(df_processado)

    # 5. DATA QUALITY -------------------------------------------------------
    logger.info("Running data quality validation")
    relatorio_qualidade = generate_data_quality_report(df_processado)
    score_qualidade = data_quality_score(relatorio_qualidade)
    logger.info("Data quality validation completed (score=%.2f/100)", score_qualidade)
    logger.info(
        "Data Quality Score improved from %.2f (RAW) to %.2f (processed) after the cleaning pipeline",
        score_qualidade_raw,
        score_qualidade,
    )

    # 6. DESCRIPTIVE STATISTICS ----------------------------------------------
    logger.info("Generating descriptive statistics")
    estatisticas_descritivas = compute_descriptive_statistics(df_processado, COLUNAS_NUMERICAS_PADRAO)

    # 7. OUTLIER DETECTION --------------------------------------------------
    logger.info("Detecting outliers (IQR)")
    outliers = detect_outliers_iqr(df_processado, COLUNAS_NUMERICAS_PADRAO)

    corr_pearson = compute_correlation_matrix(df_processado, COLUNAS_NUMERICAS_PADRAO, method="pearson")
    corr_spearman = compute_correlation_matrix(df_processado, COLUNAS_NUMERICAS_PADRAO, method="spearman")
    pares_pearson = top_correlated_pairs(df_processado, COLUNAS_NUMERICAS_PADRAO, method="pearson")
    pares_spearman = top_correlated_pairs(df_processado, COLUNAS_NUMERICAS_PADRAO, method="spearman")

    # 8. STATISTICAL TESTS ----------------------------------------------------
    logger.info("Running statistical tests")
    testes_estatisticos = run_all_statistical_tests(df_processado)

    # VISUALIZATIONS ----------------------------------------------------------
    logger.info("Generating visualizations")
    etapa_visualizacoes(df_processado, corr_pearson)

    # 9. EXPORT -----------------------------------------------------------
    logger.info("Exporting processed data and reports")
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

    data_loader.save_dataframe(df_processado, DIR_PROCESSED / "customers_clean.csv")
    data_loader.save_dataframe(relatorio_qualidade_raw, DIR_OUTPUT / "data_quality_report_raw.csv")
    data_loader.save_dataframe(relatorio_qualidade, DIR_OUTPUT / "data_quality_report.csv")
    data_loader.save_dataframe(
        estatisticas_descritivas.reset_index(), DIR_OUTPUT / "descriptive_statistics.csv"
    )
    data_loader.save_dataframe(corr_pearson.reset_index(), DIR_OUTPUT / "correlation_matrix.csv")
    data_loader.save_dataframe(corr_spearman.reset_index(), DIR_OUTPUT / "correlation_matrix_spearman.csv")
    data_loader.save_dataframe(pares_pearson, DIR_OUTPUT / "top_correlated_pairs_pearson.csv")
    data_loader.save_dataframe(pares_spearman, DIR_OUTPUT / "top_correlated_pairs_spearman.csv")
    data_loader.save_dataframe(outliers.reset_index(), DIR_OUTPUT / "outliers_report.csv")
    data_loader.save_dataframe(testes_estatisticos, DIR_OUTPUT / "statistical_tests.csv")
    data_loader.save_dataframe(candidatos_fuzzy, DIR_OUTPUT / "fuzzy_duplicates_report.csv")
    data_loader.save_dataframe(resumo_dedup, DIR_OUTPUT / "deduplication_summary.csv")
    data_loader.save_dataframe(relatorio_missing, DIR_OUTPUT / "missing_values_strategy.csv")
    data_loader.save_dataframe(
        pd.DataFrame([relatorio_idade]) if relatorio_idade else pd.DataFrame(),
        DIR_OUTPUT / "age_validation_report.csv",
    )

    duracao = time.time() - inicio
    logger.info("=" * 70)
    logger.info(
        "Pipeline completed successfully in %.1fs — %s customers, quality score=%.1f/100",
        duracao,
        len(df_processado),
        score_qualidade,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
