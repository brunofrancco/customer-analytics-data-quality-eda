"""Gera notebooks/03_descriptive_statistics.ipynb."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nb_helpers import build_notebook, code, md

cells = [
    md(
        """
# 03 · Estatística Descritiva

**Objetivo deste notebook:** calcular e interpretar um conjunto completo de
métricas descritivas para as variáveis numéricas de negócio (`age`, `income`,
`purchase_count`, `total_spent`, `satisfaction_score`), identificar outliers
pelo método IQR e analisar a correlação entre variáveis — sempre traduzindo
o resultado estatístico em significado para o negócio.

Trabalhamos a partir da base já processada em `02_data_cleaning.ipynb`.
"""
    ),
    code(
        """
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd

from src.data_loader import load_processed_data
from src.descriptive_stats import (
    COLUNAS_NUMERICAS_PADRAO, classify_distribution_shape, compute_correlation_matrix,
    compute_descriptive_statistics, detect_outliers_iqr, recommend_outlier_action,
    summarize_segments, top_correlated_pairs,
)
from src.visualization import plot_correlation_heatmap, plot_distribution

pd.set_option("display.max_columns", 25)
pd.set_option("display.width", 130)

df = load_processed_data()
df.shape
"""
    ),
    md(
        """
## Estatísticas descritivas completas

Para cada variável numérica: contagem, média, mediana, moda, desvio padrão,
variância, mínimo, máximo, amplitude, quartis (Q1/Q3), IQR, percentis
detalhados (P10-P99), assimetria (skewness), curtose (kurtosis) e coeficiente
de variação.
"""
    ),
    code(
        """
estatisticas = compute_descriptive_statistics(df, COLUNAS_NUMERICAS_PADRAO)
estatisticas.round(2).T
"""
    ),
    md(
        """
### Leitura das principais métricas

- **`age`**: média e mediana próximas (~38 anos), coeficiente de variação
  moderado — distribuição relativamente homogênea, sem assimetria forte.
- **`income`** e **`total_spent`**: coeficiente de variação **muito alto**
  (centenas de %) e curtose extremamente elevada — sinal claro de outliers
  extremos puxando a variância para cima (investigado em detalhe na seção de
  outliers, abaixo).
- **`satisfaction_score`**: variável limitada à escala 1-5, com dispersão
  naturalmente baixa (CV mais baixo entre as cinco variáveis).
"""
    ),
    md(
        """
## Forma da distribuição (assimetria e curtose)
"""
    ),
    code(
        """
for coluna in COLUNAS_NUMERICAS_PADRAO:
    skew = estatisticas.loc[coluna, "skewness"]
    kurt = estatisticas.loc[coluna, "kurtosis"]
    forma = classify_distribution_shape(skew, kurt)
    print(f"{coluna:22s} skew={skew:8.3f} ({forma['assimetria']})")
    print(f"{'':22s} kurt={kurt:8.3f} ({forma['curtose']})\\n")
"""
    ),
    md(
        """
**Impacto no negócio:** variáveis fortemente assimétricas à direita com
curtose elevada (`income`, `total_spent`, `purchase_count`) indicam que a
**média** não é uma boa medida de "cliente típico" — a mediana é mais
robusta. Isso também justifica o uso de testes não paramétricos (Spearman,
Mann-Whitney) como complemento aos testes paramétricos no notebook 04, e a
escala logarítmica nos gráficos de distribuição/dispersão dessas variáveis.
"""
    ),
    md(
        """
## Distribuições visuais
"""
    ),
    code(
        """
plot_distribution(df, "age", "Distribuição de Idade dos Clientes", "Idade (anos)")
"""
    ),
    code(
        """
plot_distribution(df, "income", "Distribuição de Renda dos Clientes", "Renda mensal (R$)")
"""
    ),
    code(
        """
plot_distribution(df, "total_spent", "Distribuição de Gasto Total dos Clientes", "Gasto total (R$)")
"""
    ),
    md(
        """
As três distribuições confirmam a leitura numérica: `age` tem formato
aproximadamente normal (sininho simétrico), enquanto `income` e `total_spent`
concentram a grande maioria dos clientes em uma faixa baixa, com uma cauda
longa à direita causada pelos outliers.
"""
    ),
    md(
        """
## Detecção de outliers (método IQR)

Valores abaixo de `Q1 - 1.5*IQR` ou acima de `Q3 + 1.5*IQR` são sinalizados
como outliers — **sem serem removidos**. A decisão de manter, investigar,
corrigir ou transformar é de negócio.
"""
    ),
    code(
        """
outliers = detect_outliers_iqr(df, COLUNAS_NUMERICAS_PADRAO)
outliers.round(2)
"""
    ),
    code(
        """
for coluna, linha in outliers.iterrows():
    recomendacao = recommend_outlier_action(linha["outlier_percentage"])
    print(f"{coluna} — {linha['outlier_count']} outliers ({linha['outlier_percentage']}%)")
    print(f"  Recomendação: {recomendacao}\\n")
"""
    ),
    md(
        """
**Interpretação de negócio:** os outliers de `income` e `total_spent`
representam uma minoria de clientes (poucos %) com valores muito acima do
padrão — plausivelmente clientes corporativos, contas de alto patrimônio, ou
possíveis erros de digitação/duplo lançamento no sistema de origem. Como o
percentual é baixo e cada caso pode ser legítimo, a recomendação é
**investigar individualmente antes de remover** — remover automaticamente
poderia descartar exatamente os clientes de maior valor para o negócio.
"""
    ),
    md(
        """
## Correlação entre variáveis

Calculamos tanto Pearson (correlação linear) quanto Spearman (correlação por
postos/monotônica), pois — como veremos — os outliers extremos identificados
acima afetam fortemente a correlação linear.
"""
    ),
    code(
        """
corr_pearson = compute_correlation_matrix(df, COLUNAS_NUMERICAS_PADRAO, method="pearson")
plot_correlation_heatmap(corr_pearson, titulo="Matriz de Correlação (Pearson)")
"""
    ),
    code(
        """
corr_spearman = compute_correlation_matrix(df, COLUNAS_NUMERICAS_PADRAO, method="spearman")
plot_correlation_heatmap(corr_spearman, titulo="Matriz de Correlação (Spearman)")
"""
    ),
    code(
        """
print("Pares mais correlacionados (Pearson):")
display(top_correlated_pairs(df, COLUNAS_NUMERICAS_PADRAO, method="pearson"))

print("\\nPares mais correlacionados (Spearman):")
display(top_correlated_pairs(df, COLUNAS_NUMERICAS_PADRAO, method="spearman"))
"""
    ),
    md(
        """
**Achado central deste notebook:** a correlação de **Pearson** entre `income`
e `total_spent` é próxima de zero, o que pareceria indicar "renda não
influencia gasto" — uma conclusão contra-intuitiva. Mas a correlação de
**Spearman** entre as mesmas variáveis é bem mais alta e estatisticamente
significativa (ver teste formal no notebook 04). A explicação: os outliers
extremos e *independentes* injetados em `income` e `total_spent` (poucos
clientes com valores absurdamente altos em uma variável, mas não
necessariamente na outra) distorcem a covariância usada por Pearson, que é
sensível a valores extremos. Spearman, por trabalhar com postos (rankings),
é robusto a essa distorção e revela a relação monotônica real que existe
entre as variáveis para a maioria dos clientes.

**Lição prática:** nunca reporte apenas Pearson em dados de negócio com
potencial de outliers — sempre calcule Spearman como verificação cruzada.

Lembrando sempre: **correlação não implica causalidade**.
"""
    ),
    md(
        """
## Prévia: comparação entre segmentos de clientes

Os segmentos (`Low / Medium / High Value`, criados no notebook 02 a partir
dos tercis de `total_spent`) serão explorados em profundidade no notebook 04,
mas uma primeira visão já é útil aqui:
"""
    ),
    code(
        """
summarize_segments(df)
"""
    ),
]

if __name__ == "__main__":
    dir_raiz = Path(__file__).resolve().parent.parent
    build_notebook(cells, str(dir_raiz / "notebooks" / "03_descriptive_statistics.ipynb"))
