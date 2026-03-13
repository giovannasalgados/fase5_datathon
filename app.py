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
# CARREGAR MODELO E DEFINIR THRESHOLD
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/modelo_risco_defasagem_mlp.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Erro ao carregar o modelo. Verifique se o scikit-learn está na versão correta (1.7.2). Erro: {e}")
    st.stop()

# O modelo foi treinado com dados altamente desbalanceados (0.15% de casos positivos)
# O threshold ótimo calculado no notebook (Índice de Youden) foi de ~0.0098 (aprox 1%)
# Qualquer probabilidade acima disso já é considerada como risco na avaliação original.
THRESHOLD_OTIMO = 0.0098

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

    # O modelo espera um DataFrame com os nomes das colunas, pois foi treinado com pipeline e imputer
    features = ["IDA", "IEG", "IPS", "IPP", "IAA", "IPV"]
    X = pd.DataFrame([[IDA, IEG, IPS, IPP, IAA, IPV]], columns=features)

    # Extrair a probabilidade bruta da classe 1 (risco)
    prob_bruta = model.predict_proba(X)[0][1]

    st.subheader("Resultado da Predição")

    # Como as probabilidades brutas são minúsculas (devido ao desbalanceamento),
    # podemos mostrar a probabilidade original, mas a classificação DEVE usar o threshold ótimo
    
    st.metric(
        label="Probabilidade calculada pelo modelo",
        value=f"{prob_bruta:.4%}"
    )
    
    st.caption(f"Nota: O modelo foi treinado em uma base onde casos de risco são raros (0.15%). O limite crítico (threshold) calculado para este modelo é de {THRESHOLD_OTIMO:.2%}.")

    # ---------------------------------------------------
    # CLASSIFICAÇÃO (Ajustada para o Threshold Ótimo)
    # ---------------------------------------------------
    
    # Criamos um "Risco Relativo" para visualização na barra de progresso (0 a 1)
    # onde o threshold (0.0098) equivale a um risco "moderado/alto"
    
    # Lógica de classificação baseada no threshold do notebook:
    if prob_bruta < (THRESHOLD_OTIMO * 0.5):
        st.success("🟢 Baixo risco de defasagem")
        nivel_risco = "Baixo"
        cor = "green"
    elif prob_bruta < THRESHOLD_OTIMO:
        st.warning("🟡 Risco moderado — atenção aos indicadores")
        nivel_risco = "Moderado"
        cor = "orange"
    else:
        st.error("🔴 Alto risco de defasagem — intervenção recomendada")
        nivel_risco = "Alto"
        cor = "red"

    st.divider()

    st.markdown(
    """
    **Interpretação**

    O modelo utiliza indicadores acadêmicos, psicossociais e
    de engajamento para estimar a probabilidade de risco.
    
    Devido ao forte desbalanceamento histórico dos dados, as probabilidades 
    brutas geradas são numericamente baixas, mas o sistema ajusta a classificação 
    automaticamente com base no ponto de corte ótimo (Threshold de Youden).
    """
    )
