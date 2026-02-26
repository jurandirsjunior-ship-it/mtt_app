# -*- coding: utf-8 -*-
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_sortables import sort_items
import re
import string
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smm
from statsmodels.formula.api import ols
import statsmodels.api as sm
import base64

# =========================
# CONFIGURAÇÃO STREAMLIT
# =========================
st.set_page_config(layout="wide")
st.title("🧪 Análise MTT - Placa 96 Poços")

# =========================
# LOGO (opcional)
# =========================

with open("logo.png", "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <style>
    .logo-container {{
        position: fixed;
        top: 15px;
        right: 30px;
        z-index: 1000;
    }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_base64}" width="120">
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# FUNÇÃO PARA LER ARQUIVO
# =========================
def ler_arquivo(arquivo):
    lines = arquivo.read().decode("utf-8").splitlines()
    start = None
    end = None

    for i, line in enumerate(lines):
        if 'Dados:' in line:
            start = i + 2
        if 'Checksum' in line:
            end = i
            break

    dados_lidos = lines[start:end]
    linhas = []

    for linha in dados_lidos:
        partes = re.split(r'\t+', linha.strip())
        if len(partes) == 13:
            letra = partes[0]
            valores = [float(v.replace(',', '.')) for v in partes[1:]]
            linhas.append([letra] + valores)

    df_raw = pd.DataFrame(linhas, columns=['Linha'] + list(range(1, 13)))
    df_raw.set_index('Linha', inplace=True)
    return df_raw

# =========================
# INICIALIZA SESSION STATE
# =========================
linhas = list(string.ascii_uppercase[:8])
colunas = list(range(1, 13))

if "placa" not in st.session_state:
    st.session_state.placa = pd.DataFrame("", index=linhas, columns=colunas)

if "grupos" not in st.session_state:
    st.session_state.grupos = []

# =========================
# ADICIONAR GRUPOS
# =========================
st.subheader("⚙️ Adicionar grupos experimentais")
novo_grupo = st.text_input("Nome do grupo")
if st.button("➕ Adicionar grupo", key="add_grupo"):
    if novo_grupo.strip() != "" and novo_grupo not in st.session_state.grupos:
        st.session_state.grupos.append(novo_grupo.strip())
        st.success("Grupo adicionado!")
    else:
        st.warning("Grupo inválido ou já existente.")

if st.session_state.grupos:
    st.write("Grupos atuais:", st.session_state.grupos)

# =========================
# UPLOAD DO ARQUIVO
# =========================
st.subheader("📂 Upload do arquivo da leitora")
arquivo = st.file_uploader("Selecione o arquivo .TXT", type="txt")

if arquivo:
    df_raw = ler_arquivo(arquivo)
    st.success("Arquivo carregado com sucesso!")

    # Heatmap
    st.subheader("📊 Heatmap da placa (Absorbância)")
    fig_heat, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_raw.astype(float), cmap="viridis_r",
                annot=True, fmt=".2f", ax=ax,
                cbar_kws={'label': 'Absorbância'})
    st.pyplot(fig_heat)

    # =========================
    # MARCAR POÇOS
    # =========================
    st.subheader("🧪 Definir grupos experimentais nos poços")
    if st.session_state.grupos:
        grupo_sel = st.selectbox("Selecione o grupo para atribuir:",
                                 st.session_state.grupos, key="select_grupo")
        for linha in linhas:
            cols = st.columns(12)
            for i, coluna in enumerate(colunas):
                poco = f"{linha}{coluna}"
                if cols[i].button(poco, key=f"poco_{linha}_{coluna}"):
                    st.session_state.placa.loc[linha, coluna] = grupo_sel
    else:
        st.info("Adicione pelo menos um grupo antes de marcar os poços.")

    st.subheader("📋 Distribuição atual")
    st.dataframe(st.session_state.placa)

    # =========================
    # ANÁLISE ESTATÍSTICA
    # =========================
    st.subheader("📊 Análise Estatística")
    if st.button("🚀 Rodar análise estatística", key="rodar_analise"):
        def get_valores(grupo):
            valores = []
            for linha in linhas:
                for coluna in colunas:
                    if st.session_state.placa.loc[linha, coluna] == grupo:
                        valores.append(df_raw.loc[linha, coluna])
            return valores

        if "Controle -" not in st.session_state.grupos:
            st.error("Adicione um grupo chamado 'Controle -'.")
            st.stop()

        controle_vals = get_valores("Controle -")
        if len(controle_vals) == 0:
            st.error("Defina pelo menos um poço como Controle -.")
            st.stop()

        media_controle = np.mean(controle_vals)
        dados = []
        for grupo in st.session_state.grupos:
            valores = get_valores(grupo)
            for v in valores:
                viab = (v / media_controle) * 100
                dados.append({"Grupo": grupo, "Viabilidade": viab})

        df_viab = pd.DataFrame(dados)
        st.session_state.df_viab = df_viab

        # ===== ANOVA =====
        modelo = ols('Viabilidade ~ C(Grupo)', data=df_viab).fit()
        anova = sm.stats.anova_lm(modelo, typ=2)
        st.subheader("📌 ANOVA")
        st.dataframe(anova)

        # ===== T-TEST + BONFERRONI =====
        controle_viab = df_viab[df_viab["Grupo"] == "Controle -"]["Viabilidade"]
        p_vals = []
        comparacoes = []
        for grupo in df_viab["Grupo"].unique():
            if grupo == "Controle -":
                continue
            grupo_vals = df_viab[df_viab["Grupo"] == grupo]["Viabilidade"]
            stat, p = ttest_ind(grupo_vals, controle_viab, equal_var=False)
            comparacoes.append(f"{grupo} vs Controle -")
            p_vals.append(p)

        if len(p_vals) > 0:
            rej, p_corr, _, _ = smm.multipletests(p_vals, alpha=0.05, method='bonferroni')
            resultados = pd.DataFrame({
                "Comparação": comparacoes,
                "p-valor bruto": p_vals,
                "p-valor corrigido": p_corr,
                "Significativo": rej
            })
            st.session_state.resultados_stats = resultados
            st.subheader("📌 Teste t (Welch) + Bonferroni")
            st.dataframe(resultados)

# =========================
# GRÁFICO (SÓ APÓS ANÁLISE)
# =========================
if "df_viab" in st.session_state:
    df_resumo = (st.session_state.df_viab
                 .groupby("Grupo")["Viabilidade"]
                 .agg(["mean", "std"])
                 .reset_index())
    st.subheader("📈 Gráfico média ± DP")
    st.markdown("### 🔄 Arraste os grupos para mudar a ordem:")
    lista_grupos = list(df_resumo["Grupo"])
    ordem_personalizada = sort_items(lista_grupos)
    df_resumo = df_resumo.set_index("Grupo").loc[ordem_personalizada].reset_index()

    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(df_resumo["Grupo"], df_resumo["mean"], yerr=df_resumo["std"], capsize=5)

    # Linha de 70%
    ax.axhline(y=70, linestyle="--", color="red", linewidth=2)

    # =========================
    # ASTERISCOS DE SIGNIFICÂNCIA
    # =========================
    if "resultados_stats" in st.session_state:
        resultados = st.session_state.resultados_stats
        for i, grupo in enumerate(df_resumo["Grupo"]):
            if grupo == "Controle -":
                continue
            linha_resultado = resultados[resultados["Comparação"] == f"{grupo} vs Controle -"]
            if not linha_resultado.empty:
                p = linha_resultado["p-valor corrigido"].values[0]
                if p < 0.001:
                    estrela = "***"
                elif p < 0.01:
                    estrela = "**"
                elif p < 0.05:
                    estrela = "*"
                else:
                    estrela = ""
                if estrela != "":
                    altura = df_resumo["mean"][i] + df_resumo["std"][i] + 5
                    ax.text(i, altura, estrela, ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax.set_ylabel("Viabilidade Celular (%)")
    ax.set_ylim(0, 150)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Salva figura no session_state
    st.session_state.figura_grafico = fig

    # =========================
    # BOTÃO DOWNLOAD
    # =========================
    st.subheader("📥 Exportar gráfico")

    formato = st.selectbox("Formato:", ["PNG", "PDF", "SVG"])

    buffer = io.BytesIO()
    st.session_state.figura_grafico.savefig(
        buffer,
        format=formato.lower(),
        dpi=300,
        bbox_inches="tight"
    )
    buffer.seek(0)

    st.download_button(
        label=f"Baixar em {formato}",
        data=buffer,
        file_name=f"grafico_MTT.{formato.lower()}",
        mime=f"image/{formato.lower()}"
    )



