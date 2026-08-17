"""Gera notebooks/01_data_profiling.ipynb."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nb_helpers import build_notebook, code, md

cells = [
    md(
        """
# 01 · Data Profiling — Diagnóstico da Base RAW

**Objetivo deste notebook:** entender, antes de qualquer limpeza, a real condição da
base de clientes consolidada a partir de diferentes sistemas de origem
(`data/raw/customers_raw.csv`). O profiling é o ponto de partida de qualquer
projeto de dados: ele orienta *quais* transformações de limpeza serão
necessárias no notebook seguinte (`02_data_cleaning.ipynb`).

Vamos analisar: dimensões, tipos de dados, valores ausentes, duplicidades,
cardinalidade e uma primeira leitura estatística (`describe`) — sempre
interpretando o que cada resultado significa para o negócio.
"""
    ),
    code(
        """
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd

from src.data_loader import load_raw_data

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

df = load_raw_data()
df.head(10)
"""
    ),
    md(
        """
## Dimensões da base

Quantos registros e colunas temos? Este número ainda **não** representa a
quantidade real de clientes — como veremos adiante, a base contém duplicidades.
"""
    ),
    code(
        """
print(f"Linhas: {df.shape[0]:,}".replace(",", "."))
print(f"Colunas: {df.shape[1]}")
"""
    ),
    md(
        """
## Tipos de dados

Como o carregamento é feito preservando tudo como texto (`dtype=str`) — decisão
proposital em `data_loader.load_raw_data`, para não perder informação antes da
limpeza — **todas** as colunas aparecem como `object`, mesmo as que deveriam
ser numéricas (`Age`, `Income`, `Purchase Count`, `Total Spent`,
`Satisfaction Score`) ou de data (`Signup Date`). Isso já é, por si só, um
problema de qualidade a ser corrigido: valores numéricos armazenados como
texto não podem ser somados, comparados ou usados em modelos estatísticos sem
conversão explícita.
"""
    ),
    code(
        """
df.dtypes
"""
    ),
    md(
        """
## Valores nulos

Contagem e percentual de valores ausentes por coluna.
"""
    ),
    code(
        """
nulos = df.isnull().sum()
percentual_nulos = (nulos / len(df) * 100).round(2)

resumo_nulos = pd.DataFrame({"missing_count": nulos, "missing_percentage": percentual_nulos})
resumo_nulos = resumo_nulos[resumo_nulos["missing_count"] > 0].sort_values("missing_count", ascending=False)
resumo_nulos
"""
    ),
    md(
        """
**Interpretação:** os valores ausentes se concentram em quatro colunas —
`Age`, `City`, `Income` e `Satisfaction Score` — com proporções entre ~4% e
~8%. Isso é consistente com uma base real vinda de múltiplos sistemas, onde
nem todo formulário de cadastro exige todos os campos. Como a proporção é
baixa a moderada, técnicas de imputação (mediana, mediana por grupo, ou
categoria "unknown") são apropriadas — **não** faremos `fillna(0)`, o que
distorceria a distribuição real dessas variáveis (ver notebook 02).
"""
    ),
    md(
        """
## Duplicidades

Duas visões: duplicidade **completa** (todas as colunas idênticas) e
duplicidade pela **chave de negócio** (`Customer ID`).
"""
    ),
    code(
        """
duplicidade_completa = df.duplicated().sum()
duplicidade_por_id = df.duplicated(subset=["Customer ID"], keep=False).sum()
ids_unicos = df["Customer ID"].nunique()

print(f"Linhas totais: {len(df):,}".replace(",", "."))
print(f"Duplicidades completas: {duplicidade_completa}")
print(f"Linhas envolvidas em duplicidade de Customer ID: {duplicidade_por_id}")
print(f"Customer IDs únicos: {ids_unicos:,}".replace(",", "."))
print(f"Diferença (linhas - IDs únicos): {len(df) - ids_unicos}")
"""
    ),
    md(
        """
**Interpretação:** a diferença entre o total de linhas e a quantidade de
`Customer ID` únicos revela quantos registros "extras" existem — uma mistura
de duplicidades completas (provavelmente reenvios do mesmo arquivo) e
duplicidades por chave (o mesmo cliente atualizado por sistemas diferentes,
com pequenas divergências). O tratamento definitivo é feito no notebook 02,
mantendo sempre o registro mais recente por `Customer ID`.

Além disso, nomes muito parecidos sob `Customer ID` **diferentes** podem
indicar o mesmo cliente cadastrado duas vezes por engano — isso não aparece
nas contagens acima e será investigado separadamente com *fuzzy matching*
(RapidFuzz) no notebook 02, sem remoção automática.
"""
    ),
    md(
        """
## Cardinalidade

Quantos valores distintos cada coluna assume — útil para identificar colunas
categóricas "sujas" (muitos valores distintos que deveriam ser poucos, como
`Gender` e `State`).
"""
    ),
    code(
        """
df.nunique().sort_values(ascending=False)
"""
    ),
    md(
        """
**Interpretação:** `Gender` deveria assumir poucas categorias (masculino,
feminino, outro), mas a cardinalidade observada é bem maior — sinal claro de
inconsistência de digitação/formato (`M`, `Masc`, `male`, `MASCULINO`...).
O mesmo vale para `State`, que deveria ter no máximo ~27 valores (as UFs
brasileiras) e apresenta uma cardinalidade maior, misturando sigla e nome
completo do estado, com e sem acentuação.
"""
    ),
    code(
        """
print("Valores distintos de Gender:", sorted(df["Gender"].dropna().unique()))
"""
    ),
    code(
        """
print("Valores distintos de State:", sorted(df["State"].dropna().unique()))
"""
    ),
    md(
        """
## Estatísticas iniciais (`describe(include="all")`)

Mesmo com os tipos ainda incorretos (texto em vez de número), o `describe`
já revela pistas importantes: a alta cardinalidade de `Income`/`Total Spent`
(quase um valor distinto por linha, esperado para variáveis contínuas) e a
presença de valores "estranhos" nas colunas numéricas armazenadas como texto.
"""
    ),
    code(
        """
df.describe(include="all").T
"""
    ),
    md(
        """
## Amostra de idades fora do intervalo válido (18-100)

Um exemplo de problema que só aparece ao olhar os *valores*, não apenas os
tipos: idades logicamente impossíveis.
"""
    ),
    code(
        """
idade_numerica = pd.to_numeric(df["Age"], errors="coerce")
idades_invalidas = df.loc[idade_numerica.notna() & ~idade_numerica.between(18, 100), "Age"]
print(f"Registros com idade fora de 18-100: {len(idades_invalidas)}")
idades_invalidas.value_counts()
"""
    ),
    md(
        """
## Resumo dos principais problemas encontrados (antes da limpeza)

Com base no profiling acima, os problemas de qualidade identificados na base
RAW são:

1. **Tipos incorretos**: colunas numéricas (`Age`, `Income`, `Purchase Count`,
   `Total Spent`, `Satisfaction Score`) e de data (`Signup Date`) armazenadas
   como texto.
2. **Valores ausentes** concentrados em `Age`, `City`, `Income` e
   `Satisfaction Score` (não em todas as colunas).
3. **Duplicidades** completas, por `Customer ID` e aproximadas (fuzzy, mesma
   pessoa sob IDs diferentes).
4. **Categorias inconsistentes** em `Gender` e `State` (múltiplas grafias
   para o mesmo valor).
5. **E-mails inconsistentes** (maiúsculas, espaços extras).
6. **Valores monetários em formato de texto** (`"4500"`, `"5.200"`,
   `"R$ 6.300,00"` — três formatos diferentes para a mesma grandeza).
7. **Datas em múltiplos formatos** (`YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`,
   `YYYY/MM/DD`).
8. **Idades inválidas** (negativas, zero ou irrealisticamente altas).
9. **Outliers extremos** em `Income`, `Total Spent` e `Purchase Count`
   (investigados estatisticamente no notebook 03).

O próximo notebook (`02_data_cleaning.ipynb`) trata cada um desses problemas
de forma isolada e documentada, usando as funções reutilizáveis do módulo
`src/data_cleaning.py`.
"""
    ),
]

if __name__ == "__main__":
    dir_raiz = Path(__file__).resolve().parent.parent
    build_notebook(cells, str(dir_raiz / "notebooks" / "01_data_profiling.ipynb"))
