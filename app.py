import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------

st.set_page_config(
    page_title="Predição de Risco de Defasagem",
    page_icon="📚",
    layout="centered"
)

# ---------------------------------------------------
# CARREGAR MODELO
# ---------------------------------------------------

model = joblib.load("models/modelo_risco_defasagem_mlp.joblib")

# ---------------------------------------------------
# TÍTULO
# ---------------------------------------------------

st.title("📚 Predição de Risco de Defasagem Educacional")

st.markdown(
"""
Aplicação baseada em **Machine Learning** desenvolvida para estimar
a probabilidade de um aluno entrar em **risco de defasagem educacional**.

Insira os indicadores educacionais do aluno para obter a estimativa de risco.
"""
)

st.divider()

# ---------------------------------------------------
# INPUT DOS INDICADORES
# ---------------------------------------------------

st.subheader("Indicadores do Aluno")

col1, col2 = st.columns(2)

with col1:
    IDA = st.slider("IDA — Desempenho Acadêmico", 0.0, 10.0, 6.0)
    IEG = st.slider("IEG — Engajamento", 0.0, 10.0, 6.0)
    IPS = st.slider("IPS — Psicossocial", 0.0, 10.0, 6.0)

with col2:
    IPP = st.slider("IPP — Psicopedagógico", 0.0, 10.0, 6.0)
    IAA = st.slider("IAA — Autoavaliação", 0.0, 10.0, 6.0)
    IPV = st.slider("IPV — Ponto de Virada", 0.0, 10.0, 6.0)

st.divider()

# ---------------------------------------------------
# PREDIÇÃO
# ---------------------------------------------------

if st.button("🔎 Calcular Probabilidade de Risco"):

    X = np.array([[IDA, IEG, IPS, IPP, IAA, IPV]])

    prob = model.predict_proba(X)[0][1]

    st.subheader("Resultado da Predição")

    st.metric(
        label="Probabilidade de risco",
        value=f"{prob:.2%}"
    )

    st.progress(float(prob))

    # ---------------------------------------------------
    # CLASSIFICAÇÃO
    # ---------------------------------------------------

    if prob < 0.30:
        st.success("🟢 Baixo risco de defasagem")

    elif prob < 0.60:
        st.warning("🟡 Risco moderado — recomenda-se acompanhamento")

    else:
        st.error("🔴 Alto risco de defasagem — intervenção recomendada")

    st.divider()

    st.markdown(
    """
    **Interpretação**

    O modelo utiliza indicadores acadêmicos, psicossociais e
    de engajamento para estimar a probabilidade de risco.

    Probabilidades mais altas indicam maior chance de
    o aluno apresentar defasagem educacional no futuro.
    """
    )