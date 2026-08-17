"""Gera notebooks/02_data_cleaning.ipynb."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nb_helpers import build_notebook, code, md

cells = [
    md(
        """
# 02 · Data Cleaning — Data Wrangling, Deduplicação e Data Quality

**Objetivo deste notebook:** aplicar, passo a passo, cada uma das transformações
de limpeza identificadas no profiling (`01_data_profiling.ipynb`), usando as
funções reutilizáveis de `src/data_cleaning.py`, `src/deduplication.py` e
`src/data_quality.py`. Ao final, geramos a base processada
(`data/processed/customers_clean.csv`) e o relatório de qualidade de dados.

Cada etapa mostra um "antes x depois" para deixar claro o efeito de cada
transformação.
"""
    ),
    code(
        """
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd

from src.data_loader import load_raw_data
from src.data_cleaning import (
    clean_email, clean_text_columns, convert_dates, convert_numeric_columns,
    handle_missing_values, standardize_column_names, standardize_customer_name,
    standardize_gender, standardize_state, validate_age,
)
from src.deduplication import (
    deduplicate_by_key, deduplication_summary, find_fuzzy_duplicates, remove_full_duplicates,
)
from src.data_quality import data_quality_score, generate_data_quality_report
from src.descriptive_stats import create_customer_segments

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

df = load_raw_data()
df.shape
"""
    ),
    md(
        """
## 1. Padronização dos nomes de colunas

`"Customer ID"` → `customer_id`, `"Total Spent"` → `total_spent`, etc. —
minúsculas, com underscore, sem espaços.
"""
    ),
    code(
        """
df = standardize_column_names(df)
df.columns.tolist()
"""
    ),
    md(
        """
## 2. Tratamento de texto (espaços) e padronização do nome do cliente
"""
    ),
    code(
        """
antes = df["customer_name"].head(5).tolist()

df = clean_text_columns(df, ["customer_name", "email", "gender", "city", "state"])
df = standardize_customer_name(df)

depois = df["customer_name"].head(5).tolist()
pd.DataFrame({"antes": antes, "depois": depois})
"""
    ),
    md(
        """
## 3. Padronização de categorias: gênero e estado

Usamos dicionários de mapeamento (`MAPA_GENERO`, `MAPA_ESTADOS_UF` em
`src/data_cleaning.py`) para consolidar todas as variantes em categorias
canônicas.
"""
    ),
    code(
        """
generos_antes = sorted(df["gender"].dropna().unique())
df = standardize_gender(df)
generos_depois = sorted(df["gender"].dropna().unique())

print(f"Categorias de gênero ANTES ({len(generos_antes)}): {generos_antes}")
print(f"Categorias de gênero DEPOIS ({len(generos_depois)}): {generos_depois}")
df["gender"].value_counts()
"""
    ),
    code(
        """
estados_antes = df["state"].nunique()
df = standardize_state(df)
estados_depois = df["state"].nunique()

print(f"Valores distintos de estado ANTES: {estados_antes}")
print(f"Valores distintos de estado DEPOIS: {estados_depois} (esperado: até 15 UFs simuladas)")
df["state"].value_counts()
"""
    ),
    md(
        """
## 4. Normalização de e-mail

Remove espaços e converte para minúsculas; também sinaliza e-mails
estruturalmente inválidos na coluna auxiliar `email_valido`.
"""
    ),
    code(
        """
df = clean_email(df)
print(f"E-mails válidos: {df['email_valido'].sum()} de {len(df)} ({df['email_valido'].mean()*100:.2f}%)")
df[["email", "email_valido"]].head(5)
"""
    ),
    md(
        """
## 5. Conversão de valores numéricos (incluindo formato monetário em R$)

`"4500"`, `"5.200"` e `"R$ 6.300,00"` devem, todos, virar o número
correspondente em ponto flutuante.
"""
    ),
    code(
        """
df = convert_numeric_columns(df)
df[["income", "purchase_count", "total_spent", "satisfaction_score", "age"]].dtypes
"""
    ),
    code(
        """
df[["income", "total_spent"]].describe()
"""
    ),
    md(
        """
## 6. Conversão de datas

Reconhece múltiplos formatos de entrada e cria colunas derivadas (ano, mês,
trimestre, dia da semana), úteis para a análise temporal no notebook 04.
"""
    ),
    code(
        """
df = convert_dates(df)
print(f"Datas não convertidas (NaT): {df['signup_date'].isna().sum()}")
df[["signup_date", "signup_date_year", "signup_date_month", "signup_date_quarter", "signup_date_weekday"]].head(5)
"""
    ),
    md(
        """
## 7. Validação de idade (18-100)

Idades fora do intervalo **não são removidas silenciosamente** — são
convertidas para ausente (`NaN`) e registradas em relatório, para posterior
imputação na etapa de tratamento de valores ausentes.
"""
    ),
    code(
        """
df, relatorio_idade = validate_age(df)
relatorio_idade
"""
    ),
    md(
        """
## 8. Deduplicação

Três níveis, do mais seguro ao mais investigativo:

1. Duplicidade completa → removida.
2. Duplicidade por `customer_id` → mantém o registro mais recente
   (`signup_date` mais alto).
3. Duplicidade aproximada (fuzzy matching por nome) → apenas reportada.
"""
    ),
    code(
        """
linhas_antes_dedup = len(df)

df, removidos_completos = remove_full_duplicates(df)
df, removidos_por_chave = deduplicate_by_key(df, key="customer_id", sort_column="signup_date")

print(f"Linhas antes da deduplicação: {linhas_antes_dedup}")
print(f"Duplicidades completas removidas: {removidos_completos}")
print(f"Duplicidades por customer_id removidas: {removidos_por_chave}")
print(f"Linhas após deduplicação determinística: {len(df)}")
print(f"customer_id únicos: {df['customer_id'].nunique()} (deve ser igual ao nº de linhas: {len(df)})")
"""
    ),
    code(
        """
# Investigação de duplicidade aproximada (fuzzy matching) — NÃO remove nada automaticamente.
candidatos_fuzzy = find_fuzzy_duplicates(df, column="customer_name", threshold=90.0)
print(f"Candidatos a duplicidade aproximada encontrados: {len(candidatos_fuzzy)}")
candidatos_fuzzy.sort_values("similarity_score", ascending=False).head(10)
"""
    ),
    md(
        """
**Interpretação:** cada linha da tabela acima é um **par candidato** — dois
`customer_id` diferentes com nomes muito parecidos (score de similaridade
≥ 90). Isso pode indicar (a) a mesma pessoa cadastrada duas vezes por engano,
ou (b) uma coincidência de nomes comuns (ex.: "Maria Silva" é um nome
frequente no Brasil). A decisão de mesclar ou não esses registros é de
negócio, não técnica — por isso o relatório é apenas investigativo e é
exportado para `data/output/fuzzy_duplicates_report.csv`, para revisão manual
por um analista.
"""
    ),
    code(
        """
resumo_dedup = deduplication_summary(linhas_antes_dedup, removidos_completos, removidos_por_chave, len(candidatos_fuzzy))
resumo_dedup
"""
    ),
    md(
        """
## 9. Tratamento de valores ausentes

Estratégia documentada por coluna — nunca `fillna(0)`:

| Coluna | Estratégia | Justificativa |
|---|---|---|
| `age` | mediana global | distribuição aproximadamente simétrica |
| `income` | mediana por estado (`state`), com fallback para mediana global | renda varia sistematicamente por região |
| `satisfaction_score` | mediana global | escala limitada (1-5), pouco sensível a outliers |
| `city` | categoria `"unknown"` | não é seguro inferir a cidade a partir de outras colunas |
"""
    ),
    code(
        """
df, relatorio_missing = handle_missing_values(df)
relatorio_missing
"""
    ),
    code(
        """
print("Valores ausentes remanescentes por coluna:")
df.isna().sum()
"""
    ),
    md(
        """
## 10. Segmentação de clientes

Criamos os segmentos de negócio (`Low / Medium / High Value`) já nesta etapa,
para que fiquem disponíveis na base processada e sejam reutilizados nos
notebooks seguintes.
"""
    ),
    code(
        """
df = create_customer_segments(df)
df["customer_segment"].value_counts()
"""
    ),
    md(
        """
## 11. Relatório de Data Quality — antes x depois

Comparamos o *Data Quality Score* calculado sobre a base bruta (apenas com
conversão numérica, para permitir os checks de validade) e sobre a base
processada final.
"""
    ),
    code(
        """
df_raw_para_dq = convert_numeric_columns(standardize_column_names(load_raw_data()))
relatorio_raw = generate_data_quality_report(df_raw_para_dq)
score_raw = data_quality_score(relatorio_raw)

relatorio_processado = generate_data_quality_report(df)
score_processado = data_quality_score(relatorio_processado)

print(f"Data Quality Score (RAW):        {score_raw:.2f} / 100")
print(f"Data Quality Score (PROCESSADO): {score_processado:.2f} / 100")
"""
    ),
    code(
        """
relatorio_raw.sort_values("value").head(10)
"""
    ),
    md(
        """
**Interpretação:** as métricas mais baixas na base RAW são exatamente as que
tratamos neste notebook — `consistency_gender_categories` e
`consistency_state_format` (categorias não padronizadas), `uniqueness_customer_id`
(duplicidades) e `validity_age`/`validity_income_positive`/`validity_satisfaction_score`
(idades inválidas e valores ausentes). Após a limpeza, o score sobe
significativamente, confirmando que as transformações aplicadas resolveram os
problemas reais identificados no profiling — e não apenas problemas
hipotéticos.
"""
    ),
    md(
        """
## 12. Exportação da base processada
"""
    ),
    code(
        """
from src.data_loader import save_dataframe, DIR_RAIZ

save_dataframe(df, DIR_RAIZ / "data" / "processed" / "customers_clean.csv")
print(f"Base processada salva com {len(df)} linhas e {df.shape[1]} colunas.")
df.head(5)
"""
    ),
    md(
        """
## Conclusão

A base processada (`customers_clean.csv`) está pronta para a análise
estatística e exploratória dos próximos notebooks: tipos corretos, categorias
padronizadas, sem duplicidades determinísticas, com valores ausentes tratados
de forma documentada e com segmentos de negócio já atribuídos. Os relatórios
de auditoria (duplicidade aproximada, estratégia de imputação, qualidade de
dados antes/depois) foram exportados para `data/output/`, garantindo
rastreabilidade de cada decisão tomada.
"""
    ),
]

if __name__ == "__main__":
    dir_raiz = Path(__file__).resolve().parent.parent
    build_notebook(cells, str(dir_raiz / "notebooks" / "02_data_cleaning.ipynb"))
