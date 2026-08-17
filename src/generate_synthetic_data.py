"""
Gerador do dataset sintético RAW utilizado neste projeto.

Este script simula uma base de clientes consolidada a partir de diferentes
sistemas de origem (CRM, e-commerce, atendimento), reproduzindo problemas de
qualidade de dados tipicamente encontrados em cenários reais:

- registros duplicados (completos, por customer_id e aproximados/fuzzy);
- valores ausentes em algumas colunas (não em todas);
- categorias inconsistentes (gênero, estado);
- e-mails inconsistentes (maiúsculas, espaços);
- valores numéricos armazenados como texto (incluindo formato monetário "R$");
- datas em múltiplos formatos;
- idades inválidas;
- outliers propositais em variáveis numéricas.

Execução:
    python -m src.generate_synthetic_data
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
N_CLIENTES_BASE = 5000
N_DUPLICATAS_COMPLETAS = 60
N_DUPLICATAS_POR_ID = 120
N_QUASE_DUPLICATAS_FUZZY = 70

DIR_RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_SAIDA = DIR_RAIZ / "data" / "raw" / "customers_raw.csv"

random.seed(SEED)
np.random.seed(SEED)
fake = Faker("pt_BR")
Faker.seed(SEED)

# ---------------------------------------------------------------------------
# Domínios de referência
# ---------------------------------------------------------------------------

# UF -> (nome completo, variantes "sujas" adicionais além do próprio UF)
ESTADOS = {
    "SP": ["São Paulo", "Sao Paulo", "sp", "SÃO PAULO"],
    "RJ": ["Rio de Janeiro", "rio de janeiro", "RJ ", "Rio De Janeiro"],
    "MG": ["Minas Gerais", "minas gerais", "mg"],
    "BA": ["Bahia", "bahia", "BA "],
    "RS": ["Rio Grande do Sul", "rio grande do sul", "rs"],
    "PR": ["Paraná", "Parana", "pr"],
    "PE": ["Pernambuco", "pernambuco", "PE "],
    "CE": ["Ceará", "Ceara", "ce"],
    "SC": ["Santa Catarina", "santa catarina", "sc"],
    "GO": ["Goiás", "Goias", "go"],
    "DF": ["Distrito Federal", "distrito federal", "df"],
    "ES": ["Espírito Santo", "Espirito Santo", "es"],
    "PA": ["Pará", "Para", "pa"],
    "AM": ["Amazonas", "amazonas", "am"],
    "MT": ["Mato Grosso", "mato grosso", "mt"],
}

CIDADES_POR_ESTADO = {
    "SP": ["São Paulo", "Campinas", "Santos", "Sorocaba", "Ribeirão Preto"],
    "RJ": ["Rio de Janeiro", "Niterói", "Petrópolis", "Duque de Caxias"],
    "MG": ["Belo Horizonte", "Uberlândia", "Juiz de Fora", "Contagem"],
    "BA": ["Salvador", "Feira de Santana", "Vitória da Conquista"],
    "RS": ["Porto Alegre", "Caxias do Sul", "Pelotas"],
    "PR": ["Curitiba", "Londrina", "Maringá"],
    "PE": ["Recife", "Olinda", "Caruaru"],
    "CE": ["Fortaleza", "Sobral", "Juazeiro do Norte"],
    "SC": ["Florianópolis", "Joinville", "Blumenau"],
    "GO": ["Goiânia", "Anápolis", "Rio Verde"],
    "DF": ["Brasília"],
    "ES": ["Vitória", "Vila Velha", "Serra"],
    "PA": ["Belém", "Ananindeua"],
    "AM": ["Manaus", "Parintins"],
    "MT": ["Cuiabá", "Várzea Grande"],
}

VARIANTES_GENERO = {
    "masculino": ["M", "Masc", "Masculino", "male", "MASCULINO", "masc"],
    "feminino": ["F", "Fem", "Feminino", "female", "FEMININO", "fem"],
    "outro": ["Outro", "Não binário", "outro", "prefer not to say", "N/I"],
}

DOMINIOS_EMAIL = ["email.com", "mail.com", "provedor.com.br", "webmail.com", "correio.net"]

FORMATOS_DATA = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]

IDADES_INVALIDAS = [-5, 0, 150, 999]


# ---------------------------------------------------------------------------
# Funções auxiliares de "sujeira" proposital
# ---------------------------------------------------------------------------

def _nome_para_email(nome: str) -> str:
    """Deriva um e-mail plausível a partir do nome completo."""

    partes = (
        nome.lower()
        .replace("á", "a").replace("à", "a").replace("ã", "a").replace("â", "a")
        .replace("é", "e").replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o").replace("ô", "o").replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
        .split()
    )
    dominio = random.choice(DOMINIOS_EMAIL)
    return f"{'.'.join(partes)}@{dominio}"


def _sujar_email(email: str) -> str:
    """Aplica inconsistências comuns de e-mail (maiúsculas, espaços)."""

    r = random.random()
    if r < 0.25:
        return email.upper()
    if r < 0.45:
        return f" {email} "
    if r < 0.55:
        return email.replace(".", ". ", 1)
    return email


def _formatar_valor_monetario(valor: float) -> str:
    """
    Formata um valor numérico como string, escolhendo aleatoriamente um dos
    formatos "sujos" descritos no briefing do projeto:
      - número puro: "4500"
      - separador de milhar com ponto, sem decimais: "5.200"
      - formato monetário brasileiro completo: "R$ 6.300,00"
    """

    valor_inteiro = int(round(valor))
    r = random.random()

    if r < 0.34:
        return str(valor_inteiro)

    if r < 0.67:
        return f"{valor_inteiro:,}".replace(",", ".")

    parte_inteira = f"{valor_inteiro:,}".replace(",", ".")
    return f"R$ {parte_inteira},00"


def _formatar_data(data) -> str:
    formato = random.choice(FORMATOS_DATA)
    return data.strftime(formato)


def _sujar_estado(uf: str) -> str:
    variantes = [uf] + ESTADOS[uf]
    return random.choice(variantes)


def _sujar_genero(categoria: str) -> str:
    return random.choice(VARIANTES_GENERO[categoria])


def _sujar_nome(nome: str) -> str:
    """Cria variações de capitalização, simulando digitação inconsistente."""

    r = random.random()
    if r < 0.15:
        return nome.upper()
    if r < 0.30:
        return nome.lower()
    return nome


# ---------------------------------------------------------------------------
# Geração da base "verdadeira" (limpa) de clientes
# ---------------------------------------------------------------------------

def _gerar_clientes_base(n: int) -> pd.DataFrame:
    registros = []
    ufs = list(ESTADOS.keys())
    pesos_uf = np.array([22, 11, 9, 6, 8, 6, 5, 5, 5, 4, 4, 3, 3, 3, 3], dtype=float)
    pesos_uf = pesos_uf / pesos_uf.sum()

    data_inicio = pd.Timestamp("2021-01-01")
    data_fim = pd.Timestamp("2025-12-31")
    intervalo_dias = (data_fim - data_inicio).days

    for i in range(1, n + 1):
        uf = np.random.choice(ufs, p=pesos_uf)
        cidade = random.choice(CIDADES_POR_ESTADO[uf])
        genero_categoria = np.random.choice(
            ["masculino", "feminino", "outro"], p=[0.47, 0.47, 0.06]
        )
        nome = fake.name_male() if genero_categoria == "masculino" else (
            fake.name_female() if genero_categoria == "feminino" else fake.name()
        )

        idade = int(np.clip(np.random.normal(38, 13), 18, 85))

        # Renda mensal (log-normal) correlacionada de leve com a idade.
        base_renda = np.random.lognormal(mean=8.1, sigma=0.45)
        renda = float(np.clip(base_renda + (idade - 38) * 25, 900, 60000))

        # Número de compras e gasto total, com correlação positiva com renda.
        n_compras = int(np.clip(np.random.poisson(6 + renda / 4000), 0, 120))
        ticket_medio = np.clip(np.random.normal(180 + renda * 0.02, 60), 20, None)
        gasto_total = float(np.round(n_compras * ticket_medio, 2))

        satisfacao = float(np.clip(np.random.normal(4.0, 0.8), 1, 5))

        data_cadastro = data_inicio + pd.Timedelta(
            days=int(np.random.triangular(0, intervalo_dias * 0.7, intervalo_dias))
        )

        registros.append(
            {
                "customer_id": f"CUST{i:06d}",
                "nome": nome,
                "genero_categoria": genero_categoria,
                "uf": uf,
                "cidade": cidade,
                "idade": idade,
                "renda": renda,
                "n_compras": n_compras,
                "gasto_total": gasto_total,
                "satisfacao": satisfacao,
                "data_cadastro": data_cadastro,
            }
        )

    return pd.DataFrame(registros)


def _renderizar_bruto(clientes: pd.DataFrame) -> pd.DataFrame:
    """Converte os registros 'limpos' em sua representação bruta (suja)."""

    linhas = []
    for _, c in clientes.iterrows():
        email_base = _nome_para_email(c["nome"])
        linhas.append(
            {
                "Customer ID": c["customer_id"],
                "Customer Name": _sujar_nome(c["nome"]),
                "Email": _sujar_email(email_base),
                "Age": c["idade"],
                "Gender": _sujar_genero(c["genero_categoria"]),
                "City": c["cidade"],
                "State": _sujar_estado(c["uf"]),
                "Signup Date": _formatar_data(c["data_cadastro"]),
                "Income": _formatar_valor_monetario(c["renda"]),
                "Purchase Count": c["n_compras"],
                "Total Spent": _formatar_valor_monetario(c["gasto_total"]),
                "Satisfaction Score": round(c["satisfacao"], 1),
            }
        )
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Injeção de problemas propositais
# ---------------------------------------------------------------------------

def _injetar_valores_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insere valores ausentes em proporções realistas.

    Observação de reprodutibilidade: usamos ``random_state`` fixos e
    explícitos por coluna (em vez de ``hash(coluna)``), pois o hash de
    strings do Python é aleatorizado por processo (PYTHONHASHSEED) — usar
    ``hash()`` aqui quebraria a reprodutibilidade determinística do dataset
    entre execuções diferentes.
    """

    df = df.copy()
    configuracao = [
        ("Age", 0.04, 501),
        ("Income", 0.06, 502),
        ("City", 0.05, 503),
        ("Satisfaction Score", 0.08, 504),
    ]
    for coluna, proporcao, semente in configuracao:
        idx = df.sample(frac=proporcao, random_state=semente).index
        df.loc[idx, coluna] = np.nan
    return df


def _injetar_idades_invalidas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx_validos = df["Age"].dropna().index
    idx = np.random.choice(idx_validos, size=int(len(df) * 0.015), replace=False)
    df.loc[idx, "Age"] = np.random.choice(IDADES_INVALIDAS, size=len(idx))
    return df


def _injetar_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Insere outliers extremos e identificáveis em variáveis numéricas."""

    df = df.copy()

    idx_income = df.sample(frac=0.01, random_state=101).index
    for i in idx_income:
        valor_extremo = np.random.uniform(150_000, 400_000)
        df.loc[i, "Income"] = _formatar_valor_monetario(valor_extremo)

    idx_spent = df.sample(frac=0.012, random_state=202).index
    for i in idx_spent:
        valor_extremo = np.random.uniform(80_000, 250_000)
        df.loc[i, "Total Spent"] = _formatar_valor_monetario(valor_extremo)

    idx_purchase = df.sample(frac=0.008, random_state=303).index
    df.loc[idx_purchase, "Purchase Count"] = np.random.randint(300, 900, size=len(idx_purchase))

    return df


def _criar_duplicatas_completas(df: pd.DataFrame, n: int) -> pd.DataFrame:
    amostra = df.sample(n=n, random_state=7)
    return amostra.copy()


def _criar_duplicatas_por_id(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Cria um segundo registro para o mesmo customer_id, simulando uma
    atualização vinda de outro sistema (com pequenas diferenças e data de
    cadastro mais recente).
    """

    amostra = df.sample(n=n, random_state=13).copy()
    for idx in amostra.index:
        valor_bruto = str(amostra.loc[idx, "Signup Date"])
        data_original = None
        for formato in FORMATOS_DATA:
            try:
                data_original = pd.to_datetime(valor_bruto, format=formato)
                break
            except (ValueError, TypeError):
                continue
        if data_original is None or pd.isna(data_original):
            data_original = pd.Timestamp("2024-01-01")
        nova_data = data_original + pd.Timedelta(days=random.randint(10, 400))
        amostra.loc[idx, "Signup Date"] = _formatar_data(nova_data)

        if pd.notna(amostra.loc[idx, "Purchase Count"]):
            amostra.loc[idx, "Purchase Count"] = int(amostra.loc[idx, "Purchase Count"]) + random.randint(1, 10)

    return amostra


def _criar_quase_duplicatas_fuzzy(clientes_base: pd.DataFrame, n: int, id_inicial: int) -> pd.DataFrame:
    """
    Cria registros de clientes 'diferentes' (novo customer_id) mas com nomes
    muito parecidos aos de clientes já existentes — candidatos a duplicidade
    aproximada (fuzzy matching), que não deve ser removida automaticamente.
    """

    amostra = clientes_base.sample(n=n, random_state=21).copy()
    linhas = []
    for offset, (_, c) in enumerate(amostra.iterrows()):
        novo_id = f"CUST{id_inicial + offset:06d}"
        variacoes_nome = [
            c["nome"].upper(),
            c["nome"].lower(),
            c["nome"].replace("da ", "").replace("de ", ""),
        ]
        nome_variado = random.choice(variacoes_nome)
        email = _sujar_email(_nome_para_email(nome_variado))

        linhas.append(
            {
                "Customer ID": novo_id,
                "Customer Name": nome_variado,
                "Email": email,
                "Age": int(np.clip(c["idade"] + random.randint(-1, 1), 18, 85)),
                "Gender": _sujar_genero(c["genero_categoria"]),
                "City": c["cidade"],
                "State": _sujar_estado(c["uf"]),
                "Signup Date": _formatar_data(c["data_cadastro"] + pd.Timedelta(days=random.randint(1, 60))),
                "Income": _formatar_valor_monetario(c["renda"] * random.uniform(0.95, 1.05)),
                "Purchase Count": max(0, c["n_compras"] + random.randint(-2, 2)),
                "Total Spent": _formatar_valor_monetario(c["gasto_total"] * random.uniform(0.9, 1.1)),
                "Satisfaction Score": round(float(np.clip(c["satisfacao"] + random.uniform(-0.3, 0.3), 1, 5)), 1),
            }
        )
    return pd.DataFrame(linhas)


def gerar_dataset_raw() -> pd.DataFrame:
    """Gera o dataset RAW completo e retorna o DataFrame final (não salvo)."""

    clientes_base = _gerar_clientes_base(N_CLIENTES_BASE)
    df_bruto = _renderizar_bruto(clientes_base)

    df_bruto = _injetar_valores_ausentes(df_bruto)
    df_bruto = _injetar_idades_invalidas(df_bruto)
    df_bruto = _injetar_outliers(df_bruto)

    duplicatas_completas = _criar_duplicatas_completas(df_bruto, N_DUPLICATAS_COMPLETAS)
    duplicatas_por_id = _criar_duplicatas_por_id(df_bruto, N_DUPLICATAS_POR_ID)
    quase_duplicatas = _criar_quase_duplicatas_fuzzy(
        clientes_base, N_QUASE_DUPLICATAS_FUZZY, id_inicial=N_CLIENTES_BASE + 1
    )

    df_final = pd.concat(
        [df_bruto, duplicatas_completas, duplicatas_por_id, quase_duplicatas],
        ignore_index=True,
    )

    # Embaralha as linhas para simular a chegada não ordenada de registros
    # de diferentes sistemas de origem.
    df_final = df_final.sample(frac=1.0, random_state=99).reset_index(drop=True)
    return df_final


def main() -> None:
    df = gerar_dataset_raw()
    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAMINHO_SAIDA, index=False, encoding="utf-8")
    print(f"Dataset RAW gerado com {len(df)} linhas em: {CAMINHO_SAIDA}")
    print(f"Customer IDs únicos: {df['Customer ID'].nunique()}")


if __name__ == "__main__":
    main()
