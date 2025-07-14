import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

st.set_page_config(page_title="NPS Insights Carglass", layout="wide")
st.title("📊 Análise Inteligente de Comentários NPS - CarGlass")

openai.api_key = st.secrets["OPENAI_API_KEY"]

def classificar_nps(nota):
    if nota >= 9:
        return "Promotor"
    elif nota >= 7:
        return "Neutro"
    else:
        return "Detrator"

def gerar_embeddings(textos):
    response = openai.Embedding.create(
        input=textos,
        model="text-embedding-ada-002"
    )
    return [d["embedding"] for d in response["data"]]

def sugerir_motivos_por_cluster(df_filtrado, n_clusters=8):
    textos = df_filtrado["Comentario"].astype(str).tolist()
    embeddings = gerar_embeddings(textos)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    df_filtrado["Cluster"] = labels

    representantes_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    motivos = df_filtrado.iloc[representantes_idx][["Cluster", "Comentario", "Classificacao_NPS"]].copy()
    motivos["Sugestão de Motivo"] = motivos["Comentario"].str[:120] + "..."
    return motivos[["Cluster", "Classificacao_NPS", "Sugestão de Motivo"]].sort_values("Cluster")

uploaded_file = st.file_uploader("Envie o arquivo Excel ou CSV com os comentários NPS:", type=[".xlsx", ".csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    else:
        df = pd.read_excel(uploaded_file, header=1)

    # Verifica se tem ao menos 7 colunas antes de renomear
    if len(df.columns) >= 7:
        col_renames = {
            df.columns[0]: "OrderId",
            df.columns[1]: "Companhia",
            df.columns[2]: "Secao",
            df.columns[3]: "Tipo_Questao",
            df.columns[4]: "Nota",
            df.columns[5]: "Motivo_Selecionado",
            df.columns[6]: "Comentario"
        }
        if len(df.columns) > 11:
            col_renames[df.columns[11]] = "Grupo_Motivo"

        df = df.rename(columns=col_renames)

        colunas_essenciais = ["OrderId", "Companhia", "Secao", "Tipo_Questao", "Nota", "Motivo_Selecionado", "Comentario"]
        if "Grupo_Motivo" in df.columns:
            colunas_essenciais.append("Grupo_Motivo")

        df = df[colunas_essenciais]
        df = df.dropna(subset=["Comentario"])
        df = df[pd.to_numeric(df["Nota"], errors="coerce").notnull()]
        df["Nota"] = df["Nota"].astype(int)
        df["Classificacao_NPS"] = df["Nota"].apply(classificar_nps)

        st.subheader("🔍 Filtros e Visualização Inicial")
        col1, col2 = st.columns(2)
        tipo_questao = col1.selectbox("Filtrar por tipo de questão:", ["Todos"] + sorted(df["Tipo_Questao"].dropna().unique().tolist()))
        nps_filter = col2.selectbox("Filtrar por tipo de NPS:", ["Todos", "Promotor", "Neutro", "Detrator"])

        df_filtrado = df.copy()
        if tipo_questao != "Todos":
            df_filtrado = df_filtrado[df_filtrado["Tipo_Questao"] == tipo_questao]
        if nps_filter != "Todos":
            df_filtrado = df_filtrado[df_filtrado["Classificacao_NPS"] == nps_filter]

        st.dataframe(df_filtrado[["Nota", "Classificacao_NPS", "Tipo_Questao", "Comentario", "Motivo_Selecionado"]], use_container_width=True)

        if st.button("🔗 Gerar nova sugestão de motivos por IA"):
            with st.spinner("Analisando comentários com IA..."):
                sugestoes = sugerir_motivos_por_cluster(df_filtrado)
                st.success("Sugestões geradas!")
                st.subheader("🎯 Sugestões de novos motivos para escolha do cliente")
                st.dataframe(sugestoes)

                csv = sugestoes.to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 Baixar sugestões em CSV", csv, "sugestoes_motivos_nps.csv", mime="text/csv")
    else:
        st.error("O arquivo enviado possui menos colunas do que o esperado. Verifique o layout.")
