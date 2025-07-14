import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import io

# Configuração da página
st.set_page_config(page_title="NPS Insights Carglass", layout="wide")
st.title("📊 Análise Inteligente de Comentários NPS - CarGlass")

# Configuração da API OpenAI
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except:
    st.error("🔑 Chave da API OpenAI não encontrada. Verifique o arquivo .streamlit/secrets.toml")
    st.stop()

def classificar_nps(nota):
    """Classifica a nota NPS em Promotor, Neutro ou Detrator"""
    try:
        nota = float(nota)
        if nota >= 9:
            return "Promotor"
        elif nota >= 7:
            return "Neutro"
        else:
            return "Detrator"
    except (ValueError, TypeError):
        return "Não classificado"

def gerar_embeddings(textos):
    """Gera embeddings dos textos usando OpenAI"""
    try:
        # Limita o tamanho dos textos para evitar problemas com a API
        textos_limitados = [str(texto)[:8000] if len(str(texto)) > 8000 else str(texto) for texto in textos]
        
        # Nova sintaxe da OpenAI API v1.0+
        response = client.embeddings.create(
            input=textos_limitados,
            model="text-embedding-ada-002"
        )
        
        return [embedding.embedding for embedding in response.data]
        
    except Exception as e:
        st.error(f"Erro ao gerar embeddings: {str(e)}")
        return None

def sugerir_motivos_por_cluster(df_filtrado, n_clusters=8):
    """Analisa os comentários e sugere novos motivos baseados em clustering"""
    try:
        # Filtra comentários válidos
        df_valido = df_filtrado.copy()
        df_valido = df_valido.dropna(subset=["Comentario"])
        
        # Converte comentários para string e filtra válidos
        df_valido["Comentario"] = df_valido["Comentario"].astype(str)
        mask_validos = (
            (df_valido["Comentario"] != '') & 
            (df_valido["Comentario"] != 'nan') &
            (df_valido["Comentario"].str.len() > 10)
        )
        df_valido = df_valido[mask_validos]
        
        if len(df_valido) < 2:
            st.error("❌ Número insuficiente de comentários válidos para análise.")
            return None
        
        if len(df_valido) < n_clusters:
            st.warning(f"Número insuficiente de comentários válidos ({len(df_valido)}). Reduzindo clusters para {max(2, len(df_valido)//2)}")
            n_clusters = max(2, len(df_valido)//2)
        
        textos = df_valido["Comentario"].astype(str).tolist()
        
        # Gera embeddings
        with st.spinner("Gerando embeddings com OpenAI..."):
            embeddings = gerar_embeddings(textos)
        
        if embeddings is None:
            return None
        
        # Aplica clustering
        with st.spinner("Realizando análise de clusters..."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
        
        df_valido = df_valido.copy()  # Evita SettingWithCopyWarning
        df_valido.loc[:, "Cluster"] = labels
        
        # Encontra comentários representativos
        representantes_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
        
        motivos = []
        for i, idx in enumerate(representantes_idx):
            try:
                linha = df_valido.iloc[idx]
                comentario_repr = linha["Comentario"]
                classificacao = linha["Classificacao_NPS"]
                
                # Cria sugestão de motivo mais concisa
                comentario_str = str(comentario_repr)
                if len(comentario_str) > 120:
                    sugestao = comentario_str[:117] + "..."
                else:
                    sugestao = comentario_str
                
                motivos.append({
                    "Cluster": f"Grupo {i+1}",
                    "Classificacao_NPS": classificacao,
                    "Comentário Representativo": sugestao,
                    "Quantidade de Comentários": len(df_valido[df_valido["Cluster"] == i])
                })
            except Exception as e:
                st.warning(f"Erro ao processar cluster {i}: {str(e)}")
                continue
        
        if not motivos:
            st.error("❌ Não foi possível gerar motivos representativos.")
            return None
            
        return pd.DataFrame(motivos).sort_values("Cluster")
    
    except Exception as e:
        st.error(f"Erro na análise de clusters: {str(e)}")
        return None

def detectar_problemas_csv(df):
    """Detecta e corrige problemas comuns em arquivos CSV"""
    problemas = []
    
    # Verifica colunas duplicadas
    if df.columns.duplicated().any():
        problemas.append("Colunas duplicadas detectadas")
        df = df.loc[:, ~df.columns.duplicated()]
    
    # Verifica se há muitas colunas vazias
    colunas_vazias = df.columns[df.isnull().all()].tolist()
    if colunas_vazias:
        problemas.append(f"Colunas completamente vazias: {len(colunas_vazias)}")
        df = df.drop(columns=colunas_vazias)
    
    # Remove linhas completamente vazias
    linhas_antes = len(df)
    df = df.dropna(how='all')
    linhas_removidas = linhas_antes - len(df)
    if linhas_removidas > 0:
        problemas.append(f"Linhas vazias removidas: {linhas_removidas}")
    
    return df, problemas

def limpar_dados(df):
    """Limpa e padroniza os dados do DataFrame"""
    # Converte colunas para string para evitar problemas de tipo
    colunas_texto = ["Comentario", "Motivo_Selecionado", "Tipo_Questao"]
    for col in colunas_texto:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('nan', '')
            df[col] = df[col].replace('<NA>', '')
    
    # Limpa e converte a coluna de nota
    if "Nota" in df.columns:
        df["Nota"] = pd.to_numeric(df["Nota"], errors="coerce")
        df = df.dropna(subset=["Nota"])
        df["Nota"] = df["Nota"].astype(int)
    
    return df

# Interface principal
st.markdown("### 📁 Upload do Arquivo")
uploaded_file = st.file_uploader(
    "Envie o arquivo Excel ou CSV com os comentários NPS:", 
    type=[".xlsx", ".csv"],
    help="O arquivo deve conter as colunas: Nota, Tipo_Questao, Comentario, Motivo_Selecionado"
)

if uploaded_file:
    try:
        # Leitura do arquivo
        with st.spinner("Carregando arquivo..."):
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            else:
                df = pd.read_excel(uploaded_file, header=1)

        # Verificação e correção de problemas no CSV
        df, problemas = detectar_problemas_csv(df)
        if problemas:
            st.warning("⚠️ Problemas detectados e corrigidos:")
            for problema in problemas:
                st.write(f"• {problema}")

        # Mostra informações básicas do arquivo
        st.info(f"📁 Arquivo carregado: {len(df)} linhas x {len(df.columns)} colunas")
        
        # Exibe as primeiras colunas para debug
        with st.expander("🔍 Visualizar estrutura do arquivo"):
            st.write("**Primeiras 10 colunas:**")
            st.write(list(df.columns[:10]))
            st.write("**Amostra dos dados:**")
            st.dataframe(df.head(3))

        # Verificação de colunas mínimas
        if len(df.columns) < 7:
            st.error(f"❌ O arquivo possui apenas {len(df.columns)} colunas. São necessárias pelo menos 7 colunas.")
            st.info("📋 Estrutura esperada: OrderId, Companhia, Secao, Tipo_Questao, Nota, Motivo_Selecionado, Comentario")
            st.stop()

        # Renomeação das colunas
        try:
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

            # Renomeia apenas as colunas que existem
            df_renamed = df.rename(columns=col_renames)
            
            # Verifica se o rename funcionou
            required_cols = ["OrderId", "Companhia", "Secao", "Tipo_Questao", "Nota", "Motivo_Selecionado", "Comentario"]
            missing_cols = [col for col in required_cols if col not in df_renamed.columns]
            
            if missing_cols:
                st.error(f"❌ Colunas não encontradas após renomeação: {missing_cols}")
                st.write("**Colunas disponíveis:**", list(df.columns))
                st.stop()
            else:
                df = df_renamed
                
        except Exception as e:
            st.error(f"❌ Erro na renomeação das colunas: {str(e)}")
            st.write("**Estrutura atual do DataFrame:**")
            st.write(f"Número de colunas: {len(df.columns)}")
            st.write("Primeiras 10 colunas:", list(df.columns[:10]))
            st.stop()

        # Seleção das colunas essenciais
        try:
            colunas_essenciais = ["OrderId", "Companhia", "Secao", "Tipo_Questao", "Nota", "Motivo_Selecionado", "Comentario"]
            if "Grupo_Motivo" in df.columns:
                colunas_essenciais.append("Grupo_Motivo")

            # Verifica se todas as colunas essenciais existem
            colunas_existentes = [col for col in colunas_essenciais if col in df.columns]
            if len(colunas_existentes) < 7:
                st.error(f"❌ Nem todas as colunas essenciais foram encontradas.")
                st.write("**Colunas encontradas:**", colunas_existentes)
                st.write("**Colunas necessárias:**", colunas_essenciais[:7])
                st.stop()

            df = df[colunas_existentes]
            
        except Exception as e:
            st.error(f"❌ Erro ao selecionar colunas essenciais: {str(e)}")
            st.stop()
        
        # Limpeza dos dados
        try:
            df = limpar_dados(df)
        except Exception as e:
            st.error(f"Erro na limpeza dos dados: {str(e)}")
            st.info("Tentando continuar com dados básicos...")
            # Limpeza mínima em caso de erro
            if "Nota" in df.columns:
                df["Nota"] = pd.to_numeric(df["Nota"], errors="coerce")
                df = df.dropna(subset=["Nota"])
            if "Comentario" in df.columns:
                df["Comentario"] = df["Comentario"].astype(str)
        
        # Remove linhas sem comentários
        df = df.dropna(subset=["Comentario"])
        df = df[df["Comentario"].astype(str) != '']
        df = df[df["Comentario"].astype(str) != 'nan']
        
        # Filtra comentários com tamanho mínimo
        mask_comentarios_validos = df["Comentario"].astype(str).str.len() > 5
        df = df[mask_comentarios_validos]
        
        # Adiciona classificação NPS
        try:
            df["Classificacao_NPS"] = df["Nota"].apply(classificar_nps)
        except Exception as e:
            st.error(f"Erro ao classificar NPS: {str(e)}")
            # Fallback: classificação manual
            df["Classificacao_NPS"] = "Não classificado"
            for idx, row in df.iterrows():
                try:
                    nota = float(row["Nota"])
                    if nota >= 9:
                        df.at[idx, "Classificacao_NPS"] = "Promotor"
                    elif nota >= 7:
                        df.at[idx, "Classificacao_NPS"] = "Neutro"
                    else:
                        df.at[idx, "Classificacao_NPS"] = "Detrator"
                except:
                    df.at[idx, "Classificacao_NPS"] = "Não classificado"

        # Exibe estatísticas básicas
        st.success(f"✅ Arquivo carregado com sucesso! {len(df)} registros válidos encontrados.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total de Registros", len(df))
        with col2:
            st.metric("👍 Promotores", len(df[df["Classificacao_NPS"] == "Promotor"]))
        with col3:
            st.metric("👎 Detratores", len(df[df["Classificacao_NPS"] == "Detrator"]))

        # Filtros
        st.markdown("### 🔍 Filtros e Visualização")
        col1, col2 = st.columns(2)
        
        with col1:
            tipos_questao = ["Todos"] + sorted(df["Tipo_Questao"].dropna().unique().tolist())
            tipo_questao = st.selectbox("Filtrar por tipo de questão:", tipos_questao)
        
        with col2:
            nps_filter = st.selectbox("Filtrar por classificação NPS:", ["Todos", "Promotor", "Neutro", "Detrator"])

        # Aplicação dos filtros
        df_filtrado = df.copy()
        if tipo_questao != "Todos":
            df_filtrado = df_filtrado[df_filtrado["Tipo_Questao"] == tipo_questao]
        if nps_filter != "Todos":
            df_filtrado = df_filtrado[df_filtrado["Classificacao_NPS"] == nps_filter]

        st.info(f"📋 Exibindo {len(df_filtrado)} registros após aplicação dos filtros")

        # Exibição dos dados filtrados - convertendo para string para evitar erros do PyArrow
        try:
            df_display = df_filtrado[["Nota", "Classificacao_NPS", "Tipo_Questao", "Comentario", "Motivo_Selecionado"]].copy()
            
            # Converte todas as colunas para string para evitar problemas de tipo
            for col in df_display.columns:
                df_display[col] = df_display[col].astype(str)
            
            st.dataframe(df_display, use_container_width=True, height=400)
        except Exception as e:
            st.error(f"Erro ao exibir dados: {str(e)}")
            # Fallback: exibe sem formatação especial
            st.write(df_filtrado[["Nota", "Classificacao_NPS", "Tipo_Questao", "Comentario", "Motivo_Selecionado"]].head(100))

        # Análise com IA
        st.markdown("### 🤖 Análise com Inteligência Artificial")
        
        if len(df_filtrado) < 5:
            st.warning("⚠️ Número insuficiente de registros para análise. Mínimo: 5 registros.")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("💡 A IA irá analisar os comentários e sugerir novos motivos baseados em padrões identificados")
            with col2:
                n_clusters = st.number_input("Número de grupos:", min_value=2, max_value=15, value=8)

            if st.button("🔗 Gerar nova sugestão de motivos por IA", type="primary"):
                if len(df_filtrado) > 0:
                    sugestoes = sugerir_motivos_por_cluster(df_filtrado, n_clusters)
                    
                    if sugestoes is not None and not sugestoes.empty:
                        st.success("✅ Análise concluída!")
                        st.markdown("### 🎯 Sugestões de novos motivos")
                        
                        # Exibe as sugestões
                        st.dataframe(sugestoes, use_container_width=True)
                        
                        # Download do CSV
                        csv_buffer = io.StringIO()
                        sugestoes.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                        csv_data = csv_buffer.getvalue().encode("utf-8-sig")
                        
                        st.download_button(
                            label="📥 Baixar sugestões em CSV",
                            data=csv_data,
                            file_name="sugestoes_motivos_nps.csv",
                            mime="text/csv"
                        )
                        
                        # Insights adicionais
                        st.markdown("### 📈 Insights Principais")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🔢 Grupos Identificados", len(sugestoes))
                        with col2:
                            classificacao_mais_comum = sugestoes["Classificacao_NPS"].mode()[0]
                            st.metric("📊 Classificação Predominante", classificacao_mais_comum)
                    else:
                        st.error("❌ Não foi possível gerar as sugestões. Verifique os dados e tente novamente.")
                else:
                    st.warning("⚠️ Nenhum registro encontrado com os filtros aplicados.")

    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
        st.info("💡 Verifique se o arquivo está no formato correto e contém as colunas necessárias.")

else:
    # Instruções quando não há arquivo
    st.markdown("""
    ### 📋 Instruções de Uso
    
    1. **Faça upload** de um arquivo CSV ou Excel com os dados do NPS
    2. **Aplique filtros** por tipo de questão e classificação NPS
    3. **Execute a análise** com IA para gerar sugestões de motivos
    4. **Baixe os resultados** em formato CSV
    
    #### 📊 Estrutura Esperada do Arquivo:
    - **Coluna 1**: OrderId
    - **Coluna 2**: Companhia  
    - **Coluna 3**: Secao
    - **Coluna 4**: Tipo_Questao (ex: "Carglass", "Lojista")
    - **Coluna 5**: Nota (0-10)
    - **Coluna 6**: Motivo_Selecionado
    - **Coluna 7**: Comentario
    - **Coluna 12** (opcional): Grupo_Motivo
    """)
