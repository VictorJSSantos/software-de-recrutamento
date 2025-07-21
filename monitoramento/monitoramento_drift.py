import streamlit as st
import pandas as pd
import plotly.express as px
import os
import datetime

# Executa consolidação de logs
resultado = os.system("python monitoramento/consolidar_logs.py")

# Configuração da página
st.set_page_config(page_title="Decision Recruiter AI - Monitoramento de Drift", layout="wide")
st.title(" Monitoramento de Drift - Decision Recruiter AI")

log_path = "monitoramento/logs/06_predictions_log.csv"

# Mensagem de sucesso na consolidação
if resultado == 0 and os.path.exists(log_path):
    st.success(" Logs consolidados com sucesso.")
else:
    st.warning(" Nenhum log consolidado encontrado ou erro ao consolidar.")

try:
    df = pd.read_csv(log_path)

    # Conversão segura do timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df = df.dropna(subset=["timestamp"])  # remove linhas inválidas
    df["data"] = df["timestamp"].dt.date

    # Filtro de datas
    min_date, max_date = df["data"].min(), df["data"].max()
    data_inicial, data_final = st.date_input(" Selecione o intervalo de datas:", [min_date, max_date], min_value=min_date, max_value=max_date)

    df_filtrado = df[(df["data"] >= data_inicial) & (df["data"] <= data_final)]

    if df_filtrado.empty:
        st.warning("Nenhum dado no intervalo selecionado.")
        st.stop()

    # Métricas rápidas
    col1, col2 = st.columns(2)
    col1.metric(" Similaridade Média", f"{df_filtrado['similaridade'].mean():.2f}")
    col2.metric(" %Match Médio", f"{(df_filtrado['match'].mean() * 100):.2f}%")

    # Baseline para alerta de drift
    baseline_sim = 0.7
    sim_dia = df_filtrado.groupby("data")["similaridade"].mean().reset_index()

    dias_com_drift = sim_dia[sim_dia["similaridade"] < baseline_sim]
    if not dias_com_drift.empty:
        st.error(f" Drift detectado em {len(dias_com_drift)} dia(s)! Similaridade < {baseline_sim}")
        st.dataframe(dias_com_drift)

    # Abas com gráficos
    tab1, tab2 = st.tabs([" Similaridade por Dia", " Match por Dia"])

    with tab1:
        st.subheader(" Similaridade média por dia")
        fig_sim = px.line(sim_dia, x="data", y="similaridade", markers=True,
                          title="Similaridade Média Diária", labels={"data": "Data", "similaridade": "Similaridade"},
                          color_discrete_sequence=["#3366CC"])
        fig_sim.update_traces(mode="lines+markers")
        fig_sim.update_layout(xaxis_title="Data", yaxis_title="Similaridade")
        st.plotly_chart(fig_sim, use_container_width=True)

    with tab2:
        st.subheader(" %Percentual de Match por dia")
        df_match = df_filtrado.groupby("data")["match"].mean().reset_index()
        df_match["match"] *= 100
        fig_match = px.bar(df_match, x="data", y="match", text_auto=True,
                           title="Percentual Diário de Match (%)", labels={"data": "Data", "match": "% Match"},
                           color_discrete_sequence=["#2CA02C"])
        fig_match.update_layout(xaxis_title="Data", yaxis_title="% Match")
        st.plotly_chart(fig_match, use_container_width=True)

except FileNotFoundError:
    st.warning("** Nenhum log de predição encontrado ainda. Faça chamadas à API para gerar logs.")
except Exception as e:
    st.error(f"** Erro ao carregar logs: {e}")
