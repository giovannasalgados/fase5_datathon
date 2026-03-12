# Análise de Risco de Defasagem Educacional

Projeto desenvolvido no Datathon – Fase 5 da Pós-Tech em Data Analytics.

## Objetivo

Construir um modelo preditivo capaz de identificar alunos com risco de defasagem educacional utilizando indicadores acadêmicos e psicossociais.

## Indicadores analisados

- IAN – Índice de Adequação de Nível
- IDA – Desempenho Acadêmico
- IEG – Engajamento
- IAA – Autoavaliação
- IPS – Psicossocial
- IPP – Psicopedagógico
- IPV – Ponto de Virada
- INDE – Indicador Global

## Metodologia

1. Análise exploratória dos dados
2. Engenharia de variáveis
3. Modelagem com MLPClassifier
4. Avaliação com ROC AUC
5. Ajuste de threshold

## Executar aplicação

```bash
pip install -r requirements.txt
streamlit run app.py