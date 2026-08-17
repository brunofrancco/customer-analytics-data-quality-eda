# Customer Analytics - Data Quality & EDA


[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/brunofrancco/customer-analytics-data-quality-eda)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Projeto de portfólio ponta a ponta em Python: transforma uma base de clientes
"suja" (proveniente de múltiplos sistemas fictícios) em uma base confiável, e
então aplica estatística descritiva, testes de hipótese e análise
exploratória de dados orientada por perguntas de negócio.

## Overview

Este projeto simula um cenário comum em empresas de médio/grande porte: uma
base de clientes consolidada a partir de diferentes sistemas (CRM,
e-commerce, atendimento), cada um com seus próprios padrões de cadastro. O
resultado é uma base com duplicidades, valores ausentes, categorias
inconsistentes, formatos numéricos e de data variados, e outliers — exatamente
os problemas que um analista ou engenheiro de dados encontra no dia a dia
antes de qualquer análise ser confiável.

O projeto cobre o fluxo completo: ingestão → profiling → limpeza (data
wrangling) → deduplicação → validação de qualidade → estatística descritiva →
detecção de outliers → testes estatísticos → EDA orientada a negócio →
insights e recomendações — tudo documentado, testado e reproduzível com um
único comando (`python run_pipeline.py`).

## Business Problem

A empresa fictícia deste projeto não confia nos números da sua base de
clientes: relatórios de receita por cliente batem de forma inconsistente
entre sistemas, e o time de CRM suspeita de clientes duplicados e categorias
digitadas de formas diferentes (ex.: gênero e estado). Antes de qualquer
análise de valor de cliente, segmentação ou modelo preditivo, é preciso
responder: **quão confiável é a nossa base, e o que ela realmente nos diz
sobre o comportamento dos clientes?**

## Objectives

- Diagnosticar objetivamente a qualidade da base bruta (profiling).
- Tratar cada problema de qualidade de forma isolada, documentada e testável
  (data wrangling / cleaning).
- Deduplicar a base de forma segura (duplicidade completa e por chave) e
  investigativa (duplicidade aproximada via fuzzy matching), sem remoções
  automáticas arriscadas.
- Medir a qualidade de dados antes e depois da limpeza (Data Quality Score).
- Calcular estatística descritiva completa e detectar outliers de forma
  transparente (método IQR, sem remoção automática).
- Responder perguntas de negócio com evidência estatística formal (testes de
  hipótese), não apenas com gráficos.
- Produzir uma base processada, relatórios e visualizações prontos para
  consumo por outras áreas (BI, CRM, produto).

## Dataset

Dataset sintético gerado em `src/generate_synthetic_data.py` (seed fixa,
100% reprodutível), simulando **5.000 clientes únicos** de base, com
duplicidades propositais elevando o arquivo bruto para **5.250 registros**
em `data/raw/customers_raw.csv`.

| Coluna (bruta) | Coluna (processada) | Descrição |
|---|---|---|
| `Customer ID` | `customer_id` | Identificador único do cliente |
| `Customer Name` | `customer_name` | Nome do cliente |
| `Email` | `email` | E-mail de contato |
| `Age` | `age` | Idade em anos |
| `Gender` | `gender` | Gênero (masculino / feminino / outro) |
| `City` | `city` | Cidade |
| `State` | `state` | Estado (UF) |
| `Signup Date` | `signup_date` | Data de cadastro |
| `Income` | `income` | Renda mensal (R$) |
| `Purchase Count` | `purchase_count` | Número de compras realizadas |
| `Total Spent` | `total_spent` | Gasto total acumulado (R$) |
| `Satisfaction Score` | `satisfaction_score` | Nota de satisfação (1-5) |

Colunas derivadas criadas durante o pipeline: `email_valido`,
`signup_date_year/month/quarter/weekday`, `customer_segment`.

## Data Quality Problems

Problemas inseridos propositalmente na base bruta (todos quantificados no
notebook `01_data_profiling.ipynb` e no relatório
`data/output/data_quality_report_raw.csv`):

- **Duplicidades**: 60 linhas completamente duplicadas, 120 registros
  duplicados por `Customer ID` (simulando atualizações de outro sistema) e
  70 registros de "quase duplicidade" — mesma pessoa sob `Customer ID`
  diferente, com pequenas variações no nome (`João Silva` / `joao silva` /
  `JOÃO SILVA`).
- **Valores ausentes**: concentrados em `Age` (~4%), `City` (~5%), `Income`
  (~6%) e `Satisfaction Score` (~8%) — não em todas as colunas.
- **Categorias inconsistentes**: `Gender` com variantes como `M`, `Masc`,
  `Masculino`, `male`; `State` misturando sigla (`SP`) e nome completo, com e
  sem acentuação (`São Paulo` / `Sao Paulo` / `sp`).
- **E-mails inconsistentes**: maiúsculas e espaços extras
  (`JOAO@EMAIL.COM`, ` joao@email.com `).
- **Valores numéricos como texto**: `income`/`total_spent` em três formatos
  distintos (`"4500"`, `"5.200"`, `"R$ 6.300,00"`).
- **Datas em múltiplos formatos**: `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`,
  `YYYY/MM/DD`, todas presentes na mesma coluna.
- **Idades inválidas**: valores como `-5`, `0`, `150`, `999` (77 registros,
  ~1,5% da base).
- **Outliers extremos e identificáveis** em `income`, `total_spent` e
  `purchase_count`.

**Data Quality Score medido**: **89.4/100** na base bruta → **100.0/100**
após o pipeline de limpeza (ver `data/output/data_quality_report_raw.csv` vs.
`data_quality_report.csv`). As métricas mais críticas na base bruta eram
`consistency_gender_categories` (1.1%) e `consistency_state_format` (42.3%) —
exatamente os problemas de padronização de categorias tratados no pipeline.

## Data Wrangling

Implementado em `src/data_cleaning.py`, com funções pequenas, independentes e
testadas (`tests/test_cleaning.py`):

`standardize_column_names` → `clean_text_columns` → `standardize_customer_name`
→ `standardize_gender` → `standardize_state` → `clean_email` →
`convert_numeric_columns` → `convert_dates` → `validate_age` →
`handle_missing_values`.

Destaques de implementação:

- **Conversão monetária genérica** (`_parse_monetary_value`): reconhece os
  três formatos de entrada e converte todos para `float`.
- **Validação de idade sem remoção silenciosa**: idades fora de `[18, 100]`
  viram `NaN` (nunca são simplesmente descartadas) e são registradas em
  `data/output/age_validation_report.csv` antes de serem imputadas.
- **Imputação de ausentes documentada** (nunca `fillna(0)`):

  | Coluna | Estratégia |
  |---|---|
  | `age` | mediana global |
  | `income` | mediana por estado (`state`), com fallback para mediana global |
  | `satisfaction_score` | mediana global |
  | `city` | categoria `"unknown"` |

  (tabela real gerada em `data/output/missing_values_strategy.csv`)

## Deduplication

Implementado em `src/deduplication.py`, com três níveis:

1. **Duplicidade completa** (`remove_full_duplicates`): removida com
   segurança — 60 registros removidos.
2. **Duplicidade por `customer_id`** (`deduplicate_by_key`): mantém o
   registro mais recente por `signup_date` — 120 registros removidos.
3. **Duplicidade aproximada** (`find_fuzzy_duplicates`, via RapidFuzz +
   blocking por letra inicial normalizada): **apenas reportada**, nunca
   removida automaticamente — 679 pares candidatos exportados para
   `data/output/fuzzy_duplicates_report.csv`, para revisão manual.

Resultado: **5.070 clientes únicos** na base processada (`customer_id` 100%
único, validado em `tests/test_deduplication.py` e no relatório de Data
Quality).

## Statistical Analysis

Implementado em `src/statistical_tests.py` (`scipy.stats`), com todos os
resultados em `data/output/statistical_tests.csv` (α = 0.05):

| Teste | Relação | Estatística | p-valor | Resultado |
|---|---|---:|---:|---|
| Pearson | income × total_spent | -0.010 | 0.501 | H0 não rejeitada |
| Spearman | income × total_spent | 0.341 | <0.001 | H0 rejeitada |
| Pearson | purchase_count × total_spent | 0.012 | 0.409 | H0 não rejeitada |
| Spearman | purchase_count × total_spent | 0.748 | <0.001 | H0 rejeitada |
| Spearman | satisfaction_score × total_spent | 0.001 | 0.925 | H0 não rejeitada |
| Spearman | age × total_spent | 0.068 | <0.001 | H0 rejeitada |
| Mann-Whitney U | satisfaction (Medium vs Low) × purchase_count | — | 0.961 | H0 não rejeitada |
| ANOVA | income por customer_segment | 0.933 | 0.394 | H0 não rejeitada |
| T-Test (Welch) | total_spent por gender | -0.565 | 0.572 | H0 não rejeitada |

O achado central: **Pearson subestima a relação entre `income`/`purchase_count`
e `total_spent` por causa dos outliers extremos** injetados nessas variáveis;
**Spearman**, robusto a outliers, revela a relação monotônica real. Essa
comparação lado a lado é discutida em detalhe no notebook 03 e 04.

## EDA

O notebook `04_exploratory_data_analysis.ipynb` responde, com evidência
estatística, perguntas organizadas em quatro blocos:

- **Perfil**: quem são os clientes (idade, renda, distribuição geográfica).
- **Comportamento**: renda e frequência de compra vs. gasto total,
  comparação entre segmentos de valor.
- **Experiência**: satisfação vs. gasto e vs. frequência de compra.
- **Temporal**: evolução mensal de aquisição de clientes.

## Key Insights

**1. Pearson subestima a relação renda × gasto por causa de outliers.**
Evidence: correlação de Pearson entre `income` e `total_spent` não é
significativa (p≈0.50); a de Spearman é (rho≈0.34, p<0.001).
Business Impact: decisões baseadas só em Pearson podem concluir, errado, que
renda não importa para o gasto.

**2. Frequência de compra é o motor do gasto total, muito mais que a renda.**
Evidence: Spearman entre `purchase_count` e `total_spent` ≈ 0.75 (p<0.001) —
a correlação mais forte observada no projeto.
Business Impact: campanhas de recorrência tendem a ter mais impacto no gasto
total do que segmentação por renda.

**3. Satisfação não está associada a gasto nem a frequência de compra.**
Evidence: Spearman satisfaction × total_spent ≈ 0 (não significativo);
Mann-Whitney de purchase_count entre grupos de satisfação também não
significativo.
Business Impact: CSAT e receita devem ser tratados como KPIs independentes,
não como proxies um do outro.

**4. Sem diferença de gasto por gênero, nem de renda entre segmentos de valor.**
Evidence: t-test de gênero e ANOVA de segmento, ambos não significativos.
Business Impact: reduz risco de viés de gênero em políticas comerciais; o
"valor do cliente" não é simplesmente um proxy do poder aquisitivo.

## Business Recommendations

1. Priorizar investimento em recorrência/frequência de compra como alavanca
   de receita, dado o vínculo estatístico forte com `total_spent`.
2. Investigar manualmente os outliers de `income`/`total_spent`
   (`data/output/outliers_report.csv`) antes de decisões automatizadas de
   pricing ou crédito.
3. Tratar os 679 pares de `fuzzy_duplicates_report.csv` como fila de revisão
   manual do time de Data Quality/CRM.
4. Não usar satisfação como proxy de valor financeiro em modelos de
   priorização de clientes.
5. Adotar Spearman como correlação padrão para KPIs financeiros sujeitos a
   outliers, como complemento (não substituto) do Pearson.

## Architecture

```text
Raw (data/raw/customers_raw.csv)
 ↓
Profiling (notebooks/01_data_profiling.ipynb)
 ↓
Cleaning (src/data_cleaning.py)
 ↓
Deduplication (src/deduplication.py)
 ↓
Data Quality (src/data_quality.py)
 ↓
Statistics (src/descriptive_stats.py, src/statistical_tests.py)
 ↓
EDA (notebooks/03 e 04, src/visualization.py)
 ↓
Insights (README.md, seção Key Insights)
```

Todo o fluxo acima é executado de ponta a ponta por `run_pipeline.py`.

## Technologies

Python 3.11 · pandas · NumPy · SciPy · Matplotlib · Seaborn · RapidFuzz ·
Faker (geração do dataset sintético) · Jupyter/nbformat · pytest · GitHub
Actions (CI).

## Project Structure

```text
customer-analytics-data-quality-eda/
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── LICENSE
├── run_pipeline.py
│
├── data/
│   ├── raw/customers_raw.csv
│   ├── processed/customers_clean.csv
│   └── output/
│       ├── data_quality_report_raw.csv
│       ├── data_quality_report.csv
│       ├── descriptive_statistics.csv
│       ├── correlation_matrix.csv
│       ├── correlation_matrix_spearman.csv
│       ├── outliers_report.csv
│       ├── statistical_tests.csv
│       ├── fuzzy_duplicates_report.csv
│       ├── deduplication_summary.csv
│       ├── missing_values_strategy.csv
│       └── age_validation_report.csv
│
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_descriptive_statistics.ipynb
│   └── 04_exploratory_data_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── generate_synthetic_data.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── deduplication.py
│   ├── data_quality.py
│   ├── descriptive_stats.py
│   ├── statistical_tests.py
│   └── visualization.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_cleaning.py
│   ├── test_deduplication.py
│   └── test_quality.py
│
├── tools/               # scripts que geram os notebooks (nbformat)
│
└── reports/figures/     # 14 figuras geradas pelo pipeline
```

## How to Run

**Opção 1 — GitHub Codespaces (sem instalar nada localmente):** clique no
badge "Open in GitHub Codespaces" no topo deste README, ou em
**Code → Codespaces → Create codespace on main** no repositório. Ele abre um
VS Code na nuvem já com o projeto clonado — é só abrir um terminal e seguir
os comandos abaixo a partir do `pip install`.

**Opção 2 — Localmente:**

```bash
git clone https://github.com/brunofrancco/customer-analytics-data-quality-eda.git
cd customer-analytics-data-quality-eda

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# (Opcional) Regenerar o dataset sintético do zero:
python -m src.generate_synthetic_data

# Executar o pipeline completo (limpeza, dedup, DQ, stats, testes, figuras):
python run_pipeline.py

# Rodar os testes automatizados:
pytest
```

Os notebooks podem ser abertos diretamente com Jupyter (`jupyter lab
notebooks/`) — eles já contêm as saídas da última execução, mas também podem
ser re-executados a qualquer momento (`Kernel > Restart & Run All`).

## Results

- Base bruta: 5.250 registros → Base processada: **5.070 clientes únicos**.
- Data Quality Score: **89.4 → 100.0** (escala 0-100) após o pipeline.
- 679 pares de possível duplicidade aproximada identificados para revisão
  manual (nenhum removido automaticamente).
- 9 testes estatísticos formais executados, com achado central sobre a
  robustez de Spearman vs. Pearson em presença de outliers.
- 14 figuras profissionais geradas em `reports/figures/`, incluindo
  distribuições, boxplots (com escala log quando necessário), heatmap de
  correlação, segmentação de clientes e evolução temporal de aquisição.

## Future Improvements

- **Machine Learning**: modelo preditivo de churn e de propensão à próxima
  compra, usando as features já tratadas (`age`, `income`, `purchase_count`,
  `satisfaction_score`, `customer_segment`).
- **Clustering** (K-Means/HDBSCAN) para uma segmentação de clientes
  multivariada, complementando a segmentação atual por tercis de
  `total_spent`.
- **Dashboard interativo** (Streamlit/Power BI) consumindo diretamente os
  CSVs de `data/output/`, para consumo self-service por áreas de negócio.
- **API** de scoring de qualidade de dados, expondo `data_quality_score()`
  como serviço para validação contínua de novas cargas.
- **Deploy em nuvem**: orquestração do `run_pipeline.py` via Cloud
  Functions/Cloud Run (GCP) com o dataset processado publicado no
  **BigQuery**, permitindo consultas analíticas em escala.
- **Testes post-hoc** (Tukey HSD) para aprofundar os resultados da ANOVA
  quando aplicável a bases futuras com diferenças significativas.

---

Projeto desenvolvido como exercício de portfólio em Data Analytics/Data
Engineering, cobrindo o ciclo completo de Data Wrangling, Data Quality,
Estatística Descritiva e Inferencial, e Análise Exploratória de Dados em
Python.
