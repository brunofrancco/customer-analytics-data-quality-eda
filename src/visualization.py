"""
Módulo de Visualização.

Gera os gráficos profissionais do projeto (Matplotlib/Seaborn), utilizando
uma paleta acessível (segura para daltonismo) em tons de azul como cor
sequencial principal, com título, rótulos de eixo, legenda e unidades em
todos os gráficos. Todas as figuras são salvas em ``reports/figures/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Paleta categórica fixa (ordem não deve ser alterada — otimizada para
# distinção sob daltonismo).
PALETA_CATEGORICA = [
    "#2a78d6",  # azul
    "#eb6834",  # laranja
    "#1baf7a",  # água
    "#eda100",  # amarelo
    "#e87ba4",  # magenta
    "#008300",  # verde
    "#4a3aa7",  # violeta
    "#e34948",  # vermelho
]

# Rampa sequencial de azul (claro -> escuro).
RAMPA_SEQUENCIAL_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

COR_GRID = "#e1e0d9"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_TEXTO_PRIMARIO = "#0b0b0b"
COR_SURFACE = "#fcfcfb"

CMAP_DIVERGENTE = sns.diverging_palette(220, 10, s=75, l=50, as_cmap=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
    }
)


def _estilo_base(fig, ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COR_GRID)
    ax.spines["bottom"].set_color(COR_GRID)
    ax.tick_params(colors=COR_TEXTO_SECUNDARIO)
    ax.xaxis.label.set_color(COR_TEXTO_PRIMARIO)
    ax.yaxis.label.set_color(COR_TEXTO_PRIMARIO)
    ax.title.set_color(COR_TEXTO_PRIMARIO)
    fig.patch.set_facecolor(COR_SURFACE)
    ax.set_facecolor(COR_SURFACE)


def _salvar(fig, caminho: Path | str | None) -> None:
    if caminho is not None:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Figure saved: %s", caminho)


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    titulo: str,
    unidade_x: str,
    caminho: Path | str | None = None,
    bins: int = 40,
):
    """Histograma + KDE de uma variável numérica, com média e mediana marcadas."""

    serie = pd.to_numeric(df[column], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(serie, bins=bins, kde=True, color=RAMPA_SEQUENCIAL_AZUL[3], edgecolor=COR_SURFACE, ax=ax)

    media, mediana = serie.mean(), serie.median()
    ax.axvline(media, color=PALETA_CATEGORICA[1], linestyle="--", linewidth=2, label=f"Média: {media:,.1f}")
    ax.axvline(mediana, color=PALETA_CATEGORICA[6], linestyle=":", linewidth=2, label=f"Mediana: {mediana:,.1f}")

    ax.set_title(titulo)
    ax.set_xlabel(unidade_x)
    ax.set_ylabel("Frequência (nº de clientes)")
    ax.legend(frameon=False)
    ax.grid(axis="y", color=COR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _estilo_base(fig, ax)
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig


def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    titulo: str,
    unidade_y: str,
    caminho: Path | str | None = None,
    escala_log: bool = False,
):
    """
    Boxplot univariado, útil para visualizar dispersão e outliers.

    Quando ``escala_log=True``, o eixo Y usa escala logarítmica (symlog) —
    recomendado para variáveis monetárias fortemente assimétricas com
    outliers extremos (ex.: income, total_spent), nas quais uma escala
    linear tornaria a caixa (Q1-Q3) ilegível.
    """

    serie = pd.to_numeric(df[column], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(4.5, 5.5))
    sns.boxplot(y=serie, color=RAMPA_SEQUENCIAL_AZUL[3], width=0.35, fliersize=3, ax=ax)
    ax.set_title(titulo)
    ax.set_ylabel(unidade_y + (" — escala log" if escala_log else ""))
    ax.set_xlabel("")
    if escala_log:
        ax.set_yscale("symlog", linthresh=max(serie.median(), 1))
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=COR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _estilo_base(fig, ax)
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame, titulo: str = "Matriz de Correlação (Pearson)", caminho: Path | str | None = None):
    """Heatmap da matriz de correlação."""

    fig, ax = plt.subplots(figsize=(max(6, len(corr.columns) * 1.1), max(5, len(corr.columns) * 0.9)))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=CMAP_DIVERGENTE,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=1,
        linecolor=COR_SURFACE,
        cbar_kws={"label": "Coeficiente de correlação"},
        ax=ax,
    )
    ax.set_title(titulo)
    _estilo_base(fig, ax)
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig


def plot_customer_segmentation(
    df: pd.DataFrame,
    segment_column: str = "customer_segment",
    value_column: str = "total_spent",
    caminho: Path | str | None = None,
):
    """Boxplot comparando o gasto total entre os segmentos de clientes."""

    ordem = [s for s in ["Low Value", "Medium Value", "High Value"] if s in df[segment_column].unique()]
    cores = dict(zip(ordem, PALETA_CATEGORICA[:3]))

    # 'total_spent' é fortemente assimétrica à direita, com outliers extremos
    # propositais (ver reports/figures/outliers.png). Exibi-los aqui achataria
    # os boxplots a ponto de torná-los ilegíveis, então o eixo Y é limitado ao
    # maior "bigode superior" (Q3 + 1.5*IQR) entre os segmentos, e os pontos
    # além desse limite são omitidos SOMENTE desta visualização comparativa.
    limite_superior = (
        df.groupby(segment_column)[value_column]
        .apply(lambda s: s.quantile(0.75) + 1.5 * (s.quantile(0.75) - s.quantile(0.25)))
        .max()
    )
    limite_eixo = limite_superior * 1.15

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(
        data=df,
        x=segment_column,
        y=value_column,
        order=ordem,
        hue=segment_column,
        palette=cores,
        legend=False,
        showfliers=False,
        ax=ax,
    )
    ax.set_ylim(0, limite_eixo)

    contagens = df[segment_column].value_counts()
    rotulos = [f"{seg}\n(n={contagens.get(seg, 0)})" for seg in ordem]
    ax.set_xticks(range(len(ordem)))
    ax.set_xticklabels(rotulos)

    ax.set_title("Gasto Total por Segmento de Cliente")
    ax.set_xlabel("Segmento (tercis de total_spent)")
    ax.set_ylabel("Gasto total (R$)")
    ax.text(
        0.5, -0.24,
        "Outliers extremos omitidos desta comparação — ver reports/figures/outliers.png",
        transform=ax.transAxes, ha="center", fontsize=8.5, color=COR_TEXTO_SECUNDARIO,
    )
    ax.grid(axis="y", color=COR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _estilo_base(fig, ax)
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig


def plot_customers_by_state(df: pd.DataFrame, column: str = "state", top_n: int = 15, caminho: Path | str | None = None):
    """Gráfico de barras horizontais com a quantidade de clientes por estado (UF)."""

    contagem = df[column].value_counts().head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.4 * len(contagem))))
    ax.barh(contagem.index.astype(str), contagem.values, color=RAMPA_SEQUENCIAL_AZUL[4])
    ax.set_title("Clientes por Estado (UF)")
    ax.set_xlabel("Número de clientes")
    ax.set_ylabel("UF")
    ax.grid(axis="x", color=COR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _estilo_base(fig, ax)
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig


def plot_monthly_customers(df: pd.DataFrame, date_column: str = "signup_date", caminho: Path | str | None = None):
    """Evolução mensal da quantidade de novos clientes cadastrados (aquisição)."""

    serie_mensal = (
        pd.to_datetime(df[date_column])
        .dt.to_period("M")
        .value_counts()
        .sort_index()
    )
    serie_mensal.index = serie_mensal.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        serie_mensal.index,
        serie_mensal.values,
        color=RAMPA_SEQUENCIAL_AZUL[4],
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.fill_between(serie_mensal.index, serie_mensal.values, color=RAMPA_SEQUENCIAL_AZUL[1], alpha=0.4)

    ax.set_title("Evolução Mensal de Novos Clientes")
    ax.set_xlabel("Mês de cadastro")
    ax.set_ylabel("Novos clientes")
    ax.grid(axis="y", color=COR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _estilo_base(fig, ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig


def plot_bivariate_relationship(
    df: pd.DataFrame,
    column_x: str,
    column_y: str,
    titulo: str,
    unidade_x: str,
    unidade_y: str,
    hue_column: str | None = None,
    caminho: Path | str | None = None,
    escala_log_x: bool = False,
    escala_log_y: bool = False,
):
    """
    Gráfico de dispersão entre duas variáveis numéricas.

    Quando as variáveis possuem outliers extremos (como ``income`` e
    ``total_spent`` neste projeto), a maior parte dos pontos fica
    visualmente "amassada" perto da origem em escala linear — nesses casos,
    ``escala_log_x``/``escala_log_y`` (symlog) tornam a relação entre a
    massa de clientes "comuns" visível, sem descartar os outliers do
    gráfico.
    """

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if hue_column and hue_column in df.columns:
        categorias = [c for c in ["Low Value", "Medium Value", "High Value"] if c in df[hue_column].unique()]
        if not categorias:
            categorias = df[hue_column].dropna().unique().tolist()
        paleta = dict(zip(categorias, PALETA_CATEGORICA))
        sns.scatterplot(
            data=df,
            x=column_x,
            y=column_y,
            hue=hue_column,
            hue_order=categorias,
            palette=paleta,
            alpha=0.55,
            s=28,
            ax=ax,
            legend=True,
        )
        ax.legend(frameon=False, title=hue_column)
    else:
        ax.scatter(df[column_x], df[column_y], alpha=0.35, s=22, color=RAMPA_SEQUENCIAL_AZUL[3])

    ax.set_title(titulo)
    ax.set_xlabel(unidade_x + (" — escala log" if escala_log_x else ""))
    ax.set_ylabel(unidade_y + (" — escala log" if escala_log_y else ""))
    if escala_log_x:
        ax.set_xscale("symlog", linthresh=max(pd.to_numeric(df[column_x], errors="coerce").median(), 1))
        ax.set_xlim(left=0)
    if escala_log_y:
        ax.set_yscale("symlog", linthresh=max(pd.to_numeric(df[column_y], errors="coerce").median(), 1))
        ax.set_ylim(bottom=0)
    ax.grid(color=COR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _estilo_base(fig, ax)
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig


def plot_outliers_overview(
    df: pd.DataFrame, columns: list[str], caminho: Path | str | None = None, escala_log: bool = True
):
    """Painel de boxplots lado a lado para visualizar outliers nas principais variáveis."""

    fig, eixos = plt.subplots(1, len(columns), figsize=(3.6 * len(columns), 5))
    if len(columns) == 1:
        eixos = [eixos]

    for ax, coluna in zip(eixos, columns):
        serie = pd.to_numeric(df[coluna], errors="coerce").dropna()
        sns.boxplot(y=serie, color=RAMPA_SEQUENCIAL_AZUL[3], width=0.4, fliersize=3, ax=ax)
        ax.set_title(coluna + (" (escala log)" if escala_log else ""))
        ax.set_ylabel("")
        ax.set_xlabel("")
        if escala_log:
            ax.set_yscale("symlog", linthresh=max(serie.median(), 1))
            ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=COR_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        _estilo_base(fig, ax)

    fig.suptitle("Visão Geral de Outliers (método IQR)", fontweight="bold", color=COR_TEXTO_PRIMARIO)
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig


def plot_missing_values(df: pd.DataFrame, caminho: Path | str | None = None):
    """Gráfico de barras com o percentual de valores ausentes por coluna."""

    ausentes = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    ausentes = ausentes[ausentes > 0]
    if ausentes.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.4 * len(ausentes))))
    ax.barh(ausentes.index.astype(str), ausentes.values, color=PALETA_CATEGORICA[7])
    ax.set_title("Percentual de Valores Ausentes por Coluna (RAW)")
    ax.set_xlabel("Ausentes (%)")
    ax.grid(axis="x", color=COR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    _estilo_base(fig, ax)
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig
