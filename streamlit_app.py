import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import io
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(page_title="NPS Insights Carglass", layout="wide")
st.title("📊 Análise Inteligente de Comentários NPS - CarGlass")

# Configuração da API OpenAI
try:
    # Tenta diferentes nomes de chave que podem estar nos secrets
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    elif "openai_api_key" in st.secrets:
        api_key = st.secrets["openai_api_key"]
    elif "OPENAI_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_KEY"]
    
    if api_key is None:
        st.error("🔑 Chave da API OpenAI não encontrada. Verifique o arquivo .streamlit/secrets.toml")
        st.info("💡 Certifique-se de que a chave está definida como: OPENAI_API_KEY = 'sua_chave_aqui'")
        st.stop()
    
    client = OpenAI(api_key=api_key)
    st.success("✅ Conexão com OpenAI estabelecida!")
    
except Exception as e:
    st.error(f"🔑 Erro ao configurar OpenAI: {str(e)}")
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


def classificar_termometro_cliente(nota, comentario):
    """
    Classifica o estado emocional do cliente baseado na nota e comentário
    """
    try:
        nota = float(nota)
        comentario_lower = str(comentario).lower()
        
        # Palavras-chave para diferentes estados emocionais
        palavras_extremamente_insatisfeito = [
            'péssimo', 'horrível', 'terrível', 'inaceitável', 'revoltante', 
            'indignado', 'furioso', 'nunca mais', 'decepcionante', 'absurdo',
            'inadmissível', 'desrespeitoso', 'lamentável', 'vergonhoso'
        ]
        
        palavras_atritado = [
            'demorou', 'demora', 'atraso', 'problema', 'dificuldade', 
            'insatisfeito', 'chateado', 'irritado', 'complicado', 'difícil',
            'transtorno', 'erro', 'falha', 'ruim', 'desorganizado'
        ]
        
        palavras_feliz = [
            'excelente', 'ótimo', 'perfeito', 'maravilhoso', 'satisfeito',
            'rápido', 'eficiente', 'bom atendimento', 'parabéns', 'recomendo',
            'tudo certo', 'nota 10', 'muito bom', 'adorei', 'fantástico'
        ]
        
        # Verifica palavras no comentário
        tem_palavras_extremas = any(palavra in comentario_lower for palavra in palavras_extremamente_insatisfeito)
        tem_palavras_atrito = any(palavra in comentario_lower for palavra in palavras_atritado)
        tem_palavras_feliz = any(palavra in comentario_lower for palavra in palavras_feliz)
        
        # Lógica de classificação
        if nota <= 2 or tem_palavras_extremas:
            return "😡 Extremamente Insatisfeito"
        elif nota <= 6 or (tem_palavras_atrito and not tem_palavras_feliz):
            return "😤 Atritado"
        elif nota >= 9 or tem_palavras_feliz:
            return "😊 Feliz"
        elif nota >= 7:
            return "😐 Neutro"
        else:
            return "😤 Atritado"
            
    except Exception as e:
        return "❓ Não classificado"


def encontrar_pior_comentario_com_os(df):
    """
    Encontra o pior comentário (menor nota) e identifica a Ordem de Serviço
    """
    try:
        # Ordena por nota (crescente) para pegar o pior
        df_ordenado = df.sort_values(['Nota', 'Classificacao_NPS'], 
                                   ascending=[True, False]).reset_index(drop=True)
        
        if len(df_ordenado) == 0:
            return None, None, None, None
        
        pior_linha = df_ordenado.iloc[0]
        
        pior_nota = pior_linha['Nota']
        pior_comentario = pior_linha['Comentario']
        order_id = pior_linha.get('OrderId', 'N/A')
        classificacao_emocional = classificar_termometro_cliente(pior_nota, pior_comentario)
        
        return pior_nota, pior_comentario, order_id, classificacao_emocional
        
    except Exception as e:
        st.error(f"Erro ao encontrar pior comentário: {str(e)}")
        return None, None, None, None


def criar_termometro_visual(df_filtrado):
    """
    Cria um termômetro visual mostrando a distribuição emocional dos clientes
    """
    if len(df_filtrado) == 0:
        return None
    
    # Aplica classificação emocional para todos os registros
    df_temp = df_filtrado.copy()
    df_temp['Termometro_Cliente'] = df_temp.apply(
        lambda row: classificar_termometro_cliente(row['Nota'], row['Comentario']), 
        axis=1
    )
    
    # Conta distribuição
    distribuicao = df_temp['Termometro_Cliente'].value_counts()
    
    # Cores para cada estado emocional
    cores = {
        '😡 Extremamente Insatisfeito': '#8B0000',  # Vermelho escuro
        '😤 Atritado': '#FF6347',                   # Tomate
        '😐 Neutro': '#FFD700',                     # Dourado
        '😊 Feliz': '#32CD32'                       # Verde lima
    }
    
    # Cria gráfico de termômetro
    fig = go.Figure()
    
    # Dados para o gráfico
    labels = distribuicao.index.tolist()
    values = distribuicao.values.tolist()
    colors = [cores.get(label, '#808080') for label in labels]
    
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{val} ({val/len(df_filtrado)*100:.1f}%)' for val in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="🌡️ Termômetro Emocional dos Clientes",
        xaxis_title="Quantidade de Clientes",
        yaxis_title="Estado Emocional",
        height=400,
        showlegend=False
    )
    
    return fig, df_temp


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


def gerar_embeddings_lote(textos, tamanho_lote=20):
    """Gera embeddings em lotes menores para evitar problemas"""
    try:
        todos_embeddings = []
        total_lotes = (len(textos) - 1) // tamanho_lote + 1
        
        # Cria um container fixo para o progresso
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for i in range(0, len(textos), tamanho_lote):
            lote = textos[i:i+tamanho_lote]
            lote_atual = i // tamanho_lote + 1
            
            # Atualiza progresso sem mover a tela
            progress = lote_atual / total_lotes
            progress_bar.progress(progress)
            status_text.text(f"🤖 Processando lote {lote_atual} de {total_lotes} ({len(lote)} comentários)")
            
            response = client.embeddings.create(
                input=lote,
                model="text-embedding-ada-002"
            )
            
            lote_embeddings = [embedding.embedding for embedding in response.data]
            todos_embeddings.extend(lote_embeddings)
        
        # Limpa o progresso
        progress_container.empty()
        return todos_embeddings
        
    except Exception as e:
        st.error(f"Erro no processamento em lotes: {str(e)}")
        return None


def gerar_embeddings(textos):
    """Gera embeddings dos textos usando OpenAI"""
    try:
        # Limpa e valida os textos de forma mais rigorosa
        textos_limpos = []
        for i, texto in enumerate(textos):
            # Converte para string e limpa
            texto_str = str(texto).strip()
            
            # Remove caracteres problemáticos
            texto_limpo = texto_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            # Remove caracteres de controle
            texto_limpo = ''.join(char for char in texto_limpo if ord(char) >= 32 or char in '\n\r\t')
            
            # Limita tamanho (OpenAI tem limite de tokens)
            if len(texto_limpo) > 8000:
                texto_limpo = texto_limpo[:8000]
            
            # Só adiciona se tiver conteúdo válido
            if len(texto_limpo.strip()) >= 10:
                textos_limpos.append(texto_limpo.strip())
        
        if not textos_limpos:
            st.error("❌ Nenhum texto válido encontrado para análise.")
            return None
        
        # Container fixo para status
        status_container = st.container()
        with status_container:
            st.info(f"🔄 Iniciando processamento de {len(textos_limpos)} comentários...")
        
        # Tenta com lotes menores se houver muitos textos
        if len(textos_limpos) > 50:
            result = gerar_embeddings_lote(textos_limpos, tamanho_lote=20)
        else:
            # Para quantidades menores, processa tudo de uma vez
            try:
                response = client.embeddings.create(
                    input=textos_limpos,
                    model="text-embedding-ada-002"
                )
                result = [embedding.embedding for embedding in response.data]
                
            except Exception as api_error:
                st.warning(f"⚠️ Tentando processamento em lotes menores...")
                result = gerar_embeddings_lote(textos_limpos, tamanho_lote=10)
        
        # Atualiza status final
        status_container.empty()
        if result:
            st.success(f"✅ Embeddings gerados com sucesso para {len(result)} textos!")
        
        return result
        
    except Exception as e:
        st.error(f"Erro geral ao gerar embeddings: {str(e)}")
        return None


def sugerir_motivos_por_cluster(df_filtrado, n_clusters=8):
    """Analisa os comentários e sugere novos motivos baseados em clustering"""
    try:
        # Filtra comentários válidos
        df_valido = df_filtrado.copy()
        df_valido = df_valido.dropna(subset=["Comentario"])
        
        # Converte comentários para string e filtra válidos
        df_valido["Comentario"] = df_valido["Comentario"].astype(str)
        
        # Filtra comentários válidos com validação mais rigorosa
        mask_validos = (
            (df_valido["Comentario"] != '') & 
            (df_valido["Comentario"] != 'nan') &
            (df_valido["Comentario"] != 'None') &
            (df_valido["Comentario"].str.len() > 10) &
            (df_valido["Comentario"].str.strip() != '')
        )
        df_valido = df_valido[mask_validos].reset_index(drop=True)
        
        if len(df_valido) < 2:
            st.error("❌ Número insuficiente de comentários válidos para análise.")
            st.info("💡 Dica: Verifique se há comentários com pelo menos 10 caracteres no dataset filtrado.")
            return None, None
        
        if len(df_valido) < n_clusters:
            st.warning(f"⚠️ Comentários disponíveis ({len(df_valido)}) menor que clusters solicitados ({n_clusters}). Ajustando para {max(2, len(df_valido)//2)} clusters.")
            n_clusters = max(2, len(df_valido)//2)
        
        # Prepara lista de textos limpos
        textos = []
        indices_validos = []
        
        for idx, comentario in enumerate(df_valido["Comentario"]):
            texto_limpo = str(comentario).strip()
            if len(texto_limpo) > 10:  # Validação extra
                textos.append(texto_limpo)
                indices_validos.append(idx)
        
        if len(textos) < 2:
            st.error("❌ Nenhum comentário válido encontrado após limpeza.")
            return None, None
            
        st.info(f"📊 Analisando {len(textos)} comentários válidos em {n_clusters} grupos.")
        
        # Gera embeddings
        with st.spinner("🤖 Gerando embeddings com OpenAI..."):
            embeddings = gerar_embeddings(textos)
        
        if embeddings is None or len(embeddings) == 0:
            return None, None
        
        # Aplica clustering
        with st.spinner("🔍 Realizando análise de clusters..."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
        
        # Cria DataFrame final apenas com textos que foram processados
        df_final = df_valido.iloc[indices_validos].copy()
        df_final.loc[:, "Cluster"] = labels
        
        # Encontra comentários representativos
        representantes_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
        
        motivos = []
        for i, idx in enumerate(representantes_idx):
            try:
                linha = df_final.iloc[idx]
                comentario_repr = linha["Comentario"]
                classificacao = linha["Classificacao_NPS"]
                
                # Cria sugestão de motivo mais concisa
                comentario_str = str(comentario_repr).strip()
                if len(comentario_str) > 120:
                    sugestao = comentario_str[:117] + "..."
                else:
                    sugestao = comentario_str
                
                # Conta quantos comentários estão neste cluster
                count_cluster = len(df_final[df_final["Cluster"] == i])
                
                motivos.append({
                    "Cluster": f"Grupo {i+1}",
                    "Classificacao_NPS": classificacao,
                    "Comentário Representativo": sugestao,
                    "Quantidade de Comentários": count_cluster
                })
            except Exception as e:
                st.warning(f"⚠️ Erro ao processar cluster {i+1}: {str(e)}")
                continue
        
        if not motivos:
            st.error("❌ Não foi possível gerar motivos representativos.")
            return None, None
            
        resultado_sugestoes = pd.DataFrame(motivos).sort_values("Cluster")
        return resultado_sugestoes, df_final
    
    except Exception as e:
        st.error(f"Erro na análise de clusters: {str(e)}")
        return None, None


def gerar_dashboard_termometro(df_filtrado):
    """
    Gera um dashboard completo com termômetro emocional e análise da OS crítica
    """
    st.markdown("---")
    st.markdown("## 🌡️ Dashboard Termômetro Emocional")
    
    # Aplica classificação emocional
    df_temp = df_filtrado.copy()
    df_temp['Termometro_Cliente'] = df_temp.apply(
        lambda row: classificar_termometro_cliente(row['Nota'], row['Comentario']), 
        axis=1
    )
    
    # Cria o termômetro visual
    fig_termometro, df_com_termometro = criar_termometro_visual(df_filtrado)
    
    # Layout em colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if fig_termometro:
            st.plotly_chart(fig_termometro, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Resumo Emocional")
        
        # Métricas resumidas
        total = len(df_temp)
        
        # Conta cada tipo
        extremo = len(df_temp[df_temp['Termometro_Cliente'] == '😡 Extremamente Insatisfeito'])
        atritado = len(df_temp[df_temp['Termometro_Cliente'] == '😤 Atritado'])
        neutro = len(df_temp[df_temp['Termometro_Cliente'] == '😐 Neutro'])
        feliz = len(df_temp[df_temp['Termometro_Cliente'] == '😊 Feliz'])
        
        # Mostra métricas
        st.metric("😡 Extremamente Insatisfeito", 
                 f"{extremo}", 
                 f"{extremo/total*100:.1f}%" if total > 0 else "0%")
        
        st.metric("😤 Atritado", 
                 f"{atritado}", 
                 f"{atritado/total*100:.1f}%" if total > 0 else "0%")
        
        st.metric("😐 Neutro", 
                 f"{neutro}", 
                 f"{neutro/total*100:.1f}%" if total > 0 else "0%")
        
        st.metric("😊 Feliz", 
                 f"{feliz}", 
                 f"{feliz/total*100:.1f}%" if total > 0 else "0%")
    
    # Seção do pior comentário
    st.markdown("### 🚨 Análise do Cliente Mais Crítico")
    
    pior_nota, pior_comentario, order_id, classificacao_emocional = encontrar_pior_comentario_com_os(df_filtrado)
    
    if pior_nota is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📉 Pior Nota", f"{pior_nota}")
        
        with col2:
            st.metric("🎯 Ordem de Serviço", f"{order_id}")
        
        with col3:
            st.metric("🌡️ Estado Emocional", classificacao_emocional)
        
        # Exibe o comentário em destaque
        st.markdown("#### 💬 Comentário Mais Crítico:")
        st.error(f"**OS {order_id}:** {pior_comentario}")
        
        # Recomendações para este caso
        st.markdown("#### ⚡ Ações Recomendadas para esta OS:")
        acoes_criticas = [
            f"🔥 **URGENTE**: Contatar cliente da OS {order_id} imediatamente",
            "📞 **Ligação direta**: Gerente deve fazer contato pessoal",
            "🎯 **Resolução**: Oferecer solução imediata e compensação",
            "📝 **Follow-up**: Acompanhar satisfação após resolução",
            "🔍 **Investigação**: Analisar processo que gerou este problema"
        ]
        
        for acao in acoes_criticas:
            st.markdown(acao)
    else:
        st.info("ℹ️ Não foi possível identificar o comentário mais crítico.")
    
    return df_com_termometro


def gerar_relatorio_detalhado(df_filtrado, sugestoes, df_final):
    """Gera um relatório detalhado da análise para apresentação à diretoria"""
    
    st.markdown("---")
    st.markdown("## 📊 Relatório Executivo da Análise NPS")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_comentarios = len(df_filtrado)
        st.metric("📝 Total Analisado", f"{total_comentarios:,}")
    
    with col2:
        promotores = len(df_filtrado[df_filtrado["Classificacao_NPS"] == "Promotor"])
        perc_promotores = (promotores/total_comentarios)*100 if total_comentarios > 0 else 0
        st.metric("👍 Promotores", f"{promotores:,}", f"{perc_promotores:.1f}%")
    
    with col3:
        detratores = len(df_filtrado[df_filtrado["Classificacao_NPS"] == "Detrator"])
        perc_detratores = (detratores/total_comentarios)*100 if total_comentarios > 0 else 0
        st.metric("👎 Detratores", f"{detratores:,}", f"{perc_detratores:.1f}%")
    
    with col4:
        nps_score = perc_promotores - perc_detratores
        st.metric("🎯 NPS Score", f"{nps_score:.1f}")
    
    # Análise por cluster
    st.markdown("### 🔍 Análise Detalhada por Grupo")
    
    # Cria tabs para diferentes análises
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Resumo Grupos", "📈 Análise Quantitativa", "💬 Comentários Representativos", "📊 Distribuição"])
    
    with tab1:
        st.markdown("#### Grupos Identificados pela IA")
        
        # Adiciona mais colunas à tabela de sugestões
        sugestoes_expandidas = sugestoes.copy()
        sugestoes_expandidas["% do Total"] = (sugestoes_expandidas["Quantidade de Comentários"] / total_comentarios * 100).round(1)
        
        # Reordena as colunas
        sugestoes_display = sugestoes_expandidas[["Cluster", "Classificacao_NPS", "Quantidade de Comentários", "% do Total", "Comentário Representativo"]]
        
        st.dataframe(sugestoes_display, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("#### Distribuição por Classificação NPS")
        
        # Análise por classificação
        analise_nps = df_filtrado.groupby("Classificacao_NPS").size().reset_index(name="Quantidade")
        analise_nps["Percentual"] = (analise_nps["Quantidade"] / total_comentarios * 100).round(1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(analise_nps, hide_index=True)
        
        with col2:
            # Principais motivos atuais
            if "Motivo_Selecionado" in df_filtrado.columns:
                top_motivos = df_filtrado["Motivo_Selecionado"].value_counts().head(5)
                st.markdown("**Top 5 Motivos Atuais:**")
                for motivo, count in top_motivos.items():
                    perc = (count/total_comentarios)*100
                    st.write(f"• {motivo}: {count} ({perc:.1f}%)")
    
    with tab3:
        st.markdown("#### Comentários Mais Representativos por Grupo")
        
        for _, row in sugestoes.iterrows():
            cluster_num = int(row["Cluster"].split()[-1])
            comentarios_cluster = df_final[df_final["Cluster"] == cluster_num-1]
            
            with st.expander(f"{row['Cluster']} - {row['Classificacao_NPS']} ({row['Quantidade de Comentários']} comentários)"):
                st.write(f"**Comentário Principal:** {row['Comentário Representativo']}")
                
                # Mostra mais alguns comentários do cluster
                if len(comentarios_cluster) > 1:
                    st.write("**Outros comentários similares:**")
                    outros_comentarios = comentarios_cluster["Comentario"].head(3).tolist()
                    for i, comentario in enumerate(outros_comentarios[:3], 1):
                        if comentario != row["Comentário Representativo"]:
                            st.write(f"{i}. {str(comentario)[:150]}...")
    
    with tab4:
        st.markdown("#### Análise de Distribuição")
        
        # Cria visualização dos grupos
        # Gráfico de barras dos grupos
        fig_grupos = px.bar(
            sugestoes_expandidas, 
            x="Cluster", 
            y="Quantidade de Comentários",
            color="Classificacao_NPS",
            title="Distribuição de Comentários por Grupo",
            color_discrete_map={
                "Promotor": "#2E8B57",
                "Neutro": "#FFD700", 
                "Detrator": "#DC143C"
            }
        )
        st.plotly_chart(fig_grupos, use_container_width=True)
        
        # Insights principais
        st.markdown("#### 🎯 Principais Insights")
        
        grupo_maior = sugestoes_expandidas.loc[sugestoes_expandidas["Quantidade de Comentários"].idxmax()]
        grupo_detrator = sugestoes_expandidas[sugestoes_expandidas["Classificacao_NPS"] == "Detrator"]
        
        insights = [
            f"• **Maior grupo identificado:** {grupo_maior['Cluster']} com {grupo_maior['Quantidade de Comentários']:,} comentários ({grupo_maior['% do Total']:.1f}% do total)"
        ]
        
        # Destaque especial para grupos de detratores
        if not grupo_detrator.empty:
            # Encontra o maior grupo de detratores
            maior_detrator = grupo_detrator.loc[grupo_detrator["Quantidade de Comentários"].idxmax()]
            total_detratores_grupos = grupo_detrator["Quantidade de Comentários"].sum()
            perc_detratores_grupos = (total_detratores_grupos / total_comentarios) * 100
            
            insights.extend([
                f"• **🚨 ALERTA CRÍTICO - {maior_detrator['Cluster']}:** {maior_detrator['Quantidade de Comentários']:,} detratores ({(maior_detrator['Quantidade de Comentários']/total_comentarios)*100:.1f}%)",
                f"  └─ **Problema:** {maior_detrator['Comentário Representativo']}",
                f"• **Total de grupos de detratores:** {len(grupo_detrator)} grupos com {total_detratores_grupos:,} comentários ({perc_detratores_grupos:.1f}%)",
                f"• **Classificação predominante:** {sugestoes['Classificacao_NPS'].mode()[0]}"
            ])
        else:
            insights.append(f"• **Classificação predominante:** {sugestoes['Classificacao_NPS'].mode()[0]}")
        
        insights.append(f"• **NPS Score atual:** {nps_score:.1f} ({'Excelente' if nps_score >= 50 else 'Bom' if nps_score >= 0 else 'Ruim'})")
        
        for insight in insights:
            if "🚨 ALERTA CRÍTICO" in insight:
                st.markdown(f"<div style='background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 5px solid #f44336;'>{insight}</div>", unsafe_allow_html=True)
            else:
                st.markdown(insight)
        
        # Recomendações
        st.markdown("#### 💡 Recomendações Estratégicas")
        
        recomendacoes = [
            "📌 **Manter pontos fortes:** Reforçar práticas que geram comentários positivos nos grupos de promotores",
            "🎯 **Focar em melhorias:** Priorizar ações corretivas nos grupos de detratores identificados",
            "📊 **Monitoramento contínuo:** Implementar acompanhamento regular usando estes grupos como baseline",
            "🔄 **Atualização de motivos:** Considerar substituir motivos atuais pelos sugeridos pela IA para maior precisão"
        ]
        
        for rec in recomendacoes:
            st.markdown(rec)
    
    return sugestoes_expandidas


def gerar_analise_detratores(df_filtrado, df_final):
    """Gera análise específica e detalhada dos detratores"""
    
    detratores = df_filtrado[df_filtrado["Classificacao_NPS"] == "Detrator"]
    
    if len(detratores) == 0:
        st.info("✅ Nenhum detrator encontrado nos dados filtrados.")
        return
    
    st.markdown("---")
    st.markdown("## 🚨 Análise Crítica de Detratores")
    st.markdown("*Esta seção foca nos clientes mais insatisfeitos para ações prioritárias*")
    
    # Métricas críticas
    col1, col2, col3, col4 = st.columns(4)
    
    total_detratores = len(detratores)
    total_geral = len(df_filtrado)
    perc_detratores = (total_detratores / total_geral) * 100
    
    with col1:
        st.metric("🔥 Total Detratores", f"{total_detratores:,}", f"{perc_detratores:.1f}% do total")
    
    with col2:
        nota_media = detratores["Nota"].mean()
        st.metric("📉 Nota Média", f"{nota_media:.1f}", "Crítico se < 5")
    
    with col3:
        notas_zero = len(detratores[detratores["Nota"] == 0])
        perc_zero = (notas_zero / total_detratores) * 100 if total_detratores > 0 else 0
        st.metric("💥 Notas 0", f"{notas_zero}", f"{perc_zero:.1f}%")
    
    with col4:
        if df_final is not None and "Cluster" in df_final.columns:
            clusters_detratores = len(df_final[df_final["Classificacao_NPS"] == "Detrator"]["Cluster"].unique())
            st.metric("🎯 Grupos Detratores", clusters_detratores)
        else:
            st.metric("🎯 Grupos Detratores", "N/A")
    
    # Tabs para análise detalhada
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Principais Problemas", "📊 Distribuição", "💬 Comentários Críticos", "⚡ Plano de Ação"])
    
    with tab1:
        st.markdown("### 🎯 Principais Motivos de Insatisfação")
        
        if "Motivo_Selecionado" in detratores.columns:
            # Top problemas dos detratores
            problemas = detratores["Motivo_Selecionado"].value_counts().head(10)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Top 10 Motivos de Detratores")
                for i, (motivo, count) in enumerate(problemas.items(), 1):
                    perc = (count / total_detratores) * 100
                    
                    # Destaca o principal problema
                    if i == 1:
                        st.markdown(f"**🔴 {i}. {motivo}**")
                        st.markdown(f"   📊 **{count:,} casos ({perc:.1f}% dos detratores)**")
                        st.markdown(f"   🚨 **PRIORIDADE MÁXIMA**")
                    elif i <= 3:
                        st.markdown(f"**🟡 {i}. {motivo}**")
                        st.markdown(f"   📊 {count:,} casos ({perc:.1f}%)")
                    else:
                        st.markdown(f"{i}. {motivo}: {count:,} casos ({perc:.1f}%)")
            
            with col2:
                # Análise do problema principal
                problema_principal = problemas.index[0]
                casos_principais = detratores[detratores["Motivo_Selecionado"] == problema_principal]
                
                st.markdown("#### 🔍 Análise do Problema #1")
                st.metric("Casos", len(casos_principais))
                
                if len(casos_principais) > 0:
                    nota_media_problema = casos_principais["Nota"].mean()
                    st.metric("Nota Média", f"{nota_media_problema:.1f}")
                    
                    # Distribuição de notas do problema principal
                    distribuicao_notas = casos_principais["Nota"].value_counts().sort_index()
                    st.markdown("**Distribuição de Notas:**")
                    for nota, count in distribuicao_notas.items():
                        st.write(f"Nota {nota}: {count} casos")
    
    with tab2:
        st.markdown("### 📈 Distribuição e Tendências")
        
        # Gráfico de distribuição de notas dos detratores
        fig_notas = px.histogram(
            detratores, 
            x="Nota", 
            title="Distribuição de Notas - Detratores",
            color_discrete_sequence=["#DC143C"]
        )
        fig_notas.update_layout(xaxis_title="Nota NPS", yaxis_title="Quantidade")
        st.plotly_chart(fig_notas, use_container_width=True)
        
        # Análise por tipo de questão se disponível
        if "Tipo_Questao" in detratores.columns:
            st.markdown("#### 📋 Detratores por Tipo de Questão")
            tipo_questao = detratores["Tipo_Questao"].value_counts()
            
            fig_tipo = px.pie(
                values=tipo_questao.values,
                names=tipo_questao.index,
                title="Distribuição de Detratores por Tipo"
            )
            st.plotly_chart(fig_tipo, use_container_width=True)
    
    with tab3:
        st.markdown("### 💬 Comentários Mais Críticos")
        
        # Comentários com notas 0-2 (mais críticos)
        criticos = detratores[detratores["Nota"] <= 2].sort_values("Nota")
        
        if len(criticos) > 0:
            st.markdown(f"#### 🚨 {len(criticos)} Comentários Extremamente Críticos (Notas 0-2)")
            
            for idx, row in criticos.head(5).iterrows():
                with st.expander(f"Nota {row['Nota']} - OS: {row.get('OrderId', 'N/A')} - {row.get('Motivo_Selecionado', 'N/A')}"):
                    st.write(f"**Comentário:** {row['Comentario']}")
                    st.write(f"**Ordem de Serviço:** {row.get('OrderId', 'N/A')}")
                    if "Tipo_Questao" in row:
                        st.write(f"**Tipo:** {row['Tipo_Questao']}")
                    if "Companhia" in row:
                        st.write(f"**Seguradora:** {row['Companhia']}")
                    
                    # Mostra o termômetro emocional deste comentário
                    estado_emocional = classificar_termometro_cliente(row['Nota'], row['Comentario'])
                    st.write(f"**Estado Emocional:** {estado_emocional}")
        
        # Comentários do maior grupo de detratores (se existir análise de clusters)
        st.markdown("#### 🎯 Comentários do Maior Grupo de Detratores")
        if df_final is not None and "Cluster" in df_final.columns:
            detratores_clustered = df_final[df_final["Classificacao_NPS"] == "Detrator"]
            if len(detratores_clustered) > 0:
                maior_cluster = detratores_clustered["Cluster"].value_counts().index[0]
                comentarios_cluster = detratores_clustered[detratores_clustered["Cluster"] == maior_cluster]
                
                st.write(f"**Grupo {maior_cluster + 1} - {len(comentarios_cluster)} comentários**")
                for _, row in comentarios_cluster.head(3).iterrows():
                    st.write(f"• **OS {row.get('OrderId', 'N/A')}:** {str(row['Comentario'])[:200]}...")
        else:
            st.info("Análise de clusters não disponível para esta visualização.")
    
    with tab4:
        st.markdown("### ⚡ Plano de Ação Imediato")
        
        st.markdown("#### 🚨 Ações Prioritárias (0-30 dias)")
        
        acoes_imediatas = [
            "🎯 **Foco no problema #1**: Resolver urgentemente as dificuldades de agendamento",
            "📞 **Central de atendimento**: Reforçar treinamento da equipe para agendamentos",
            "🔧 **Sistema de agendamento**: Revisar e melhorar a plataforma/processo",
            "📱 **Múltiplos canais**: Facilitar agendamento por telefone, WhatsApp e site",
            "⏰ **Prazos claros**: Estabelecer SLA máximo para agendamento (ex: 24h)"
        ]
        
        for acao in acoes_imediatas:
            st.markdown(acao)
        
        st.markdown("#### 📈 Ações de Médio Prazo (30-90 dias)")
        
        acoes_medio_prazo = [
            "🏪 **Expansão da rede**: Avaliar abertura de lojas em regiões deficitárias",
            "🤖 **Automação**: Implementar agendamento automático com confirmação",
            "📊 **Monitoramento**: Dashboard em tempo real dos agendamentos",
            "🎓 **Treinamento**: Capacitação contínua das equipes de atendimento",
            "📞 **Callback**: Sistema de retorno automático para casos não resolvidos"
        ]
        
        for acao in acoes_medio_prazo:
            st.markdown(acao)
        
        # ROI estimado
        st.markdown("#### 💰 Impacto Financeiro Estimado")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**💸 Custo atual dos detratores:**")
            st.write(f"• {total_detratores:,} clientes insatisfeitos")
            st.write(f"• Potencial perda de receita")
            st.write(f"• Impacto na reputação da marca")
        
        with col2:
            st.markdown("**💹 Benefício da melhoria:**")
            st.write(f"• Conversão de {total_detratores//2:,} detratores em neutros")
            st.write(f"• Melhoria do NPS em ~{(total_detratores//2/total_geral)*100:.1f} pontos")
            st.write(f"• Redução de reclamações e cancelamentos")
    
    return detratores


# Interface principal
st.markdown("### 📁 Upload do Arquivo")
uploaded_file = st.file_uploader(
    "Envie o arquivo Excel ou CSV com os comentários NPS:", 
    type=[".xlsx", ".csv"],
    help="O arquivo deve conter as colunas: OrderId, Companhia, Secao, Tipo_Questao, Nota, Motivo_Selecionado, Comentario"
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
            
            if len(df.columns) > 8:
                col_renames[df.columns[8]] = "Grupo_Motivo"

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

        # === NOVO: DASHBOARD TERMÔMETRO EMOCIONAL ===
        if len(df) > 0:
            df_com_termometro = gerar_dashboard_termometro(df)

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
            df_display = df_filtrado[["OrderId", "Nota", "Classificacao_NPS", "Tipo_Questao", "Comentario", "Motivo_Selecionado"]].copy()
            
            # Converte todas as colunas para string para evitar problemas de tipo
            for col in df_display.columns:
                df_display[col] = df_display[col].astype(str)
            
            st.dataframe(df_display, use_container_width=True, height=400)
        except Exception as e:
            st.error(f"Erro ao exibir dados: {str(e)}")
            # Fallback: exibe sem formatação especial
            st.write(df_filtrado[["OrderId", "Nota", "Classificacao_NPS", "Tipo_Questao", "Comentario", "Motivo_Selecionado"]].head(100))

        # === TERMÔMETRO PARA DADOS FILTRADOS ===
        if len(df_filtrado) > 0:
            st.markdown("### 🌡️ Termômetro dos Dados Filtrados")
            df_filtrado_termometro = gerar_dashboard_termometro(df_filtrado)

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
                # Primeiro, vamos testar a conexão com a API
                st.info("🔬 Testando conexão com OpenAI...")
                try:
                    test_response = client.embeddings.create(
                        input=["Teste de conexão"],
                        model="text-embedding-ada-002"
                    )
                    st.success("✅ Conexão com OpenAI funcionando!")
                except Exception as test_error:
                    st.error(f"❌ Falha no teste de conexão: {str(test_error)}")
                    st.stop()
                
                if len(df_filtrado) > 0:
                    resultado = sugerir_motivos_por_cluster(df_filtrado, n_clusters)
                    
                    if resultado is not None and len(resultado) == 2:
                        sugestoes, df_final = resultado
                        
                        if sugestoes is not None and not sugestoes.empty:
                            # Centraliza o conteúdo
                            col_esq, col_centro, col_dir = st.columns([1, 8, 1])
                            
                            with col_centro:
                                st.success("✅ Análise concluída!")
                                st.markdown("### 🎯 Sugestões de Novos Motivos NPS")
                                
                                # Exibe as sugestões de forma mais organizada
                                st.dataframe(sugestoes, use_container_width=True, hide_index=True)
                                
                                # Download do CSV
                                csv_buffer = io.StringIO()
                                sugestoes.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                                csv_data = csv_buffer.getvalue().encode("utf-8-sig")
                                
                                # Centraliza o botão de download
                                col1, col2, col3 = st.columns([2, 4, 2])
                                with col2:
                                    st.download_button(
                                        label="📥 Baixar sugestões em CSV",
                                        data=csv_data,
                                        file_name="sugestoes_motivos_nps.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            
                            # Gera relatório detalhado (fora da coluna centralizada para usar toda a largura)
                            relatorio_expandido = gerar_relatorio_detalhado(df_filtrado, sugestoes, df_final)
                            
                            # Análise específica de detratores
                            analise_detratores = gerar_analise_detratores(df_filtrado, df_final)
                        else:
                            st.error("❌ Não foi possível gerar as sugestões. Verifique os dados e tente novamente.")
                    else:
                        st.error("❌ Erro no processamento dos clusters.")
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
    2. **Visualize o termômetro emocional** para entender o estado dos clientes
    3. **Identifique a OS mais crítica** para ação imediata
    4. **Aplique filtros** por tipo de questão e classificação NPS
    5. **Execute a análise** com IA para gerar sugestões de motivos
    6. **Baixe os resultados** em formato CSV
    
    #### 📊 Estrutura Esperada do Arquivo:
    - **Coluna 1**: OrderId (Ordem de Serviço)
    - **Coluna 2**: Companhia  
    - **Coluna 3**: Secao
    - **Coluna 4**: Tipo_Questao (ex: "Carglass", "Lojista")
    - **Coluna 5**: Nota (0-10)
    - **Coluna 6**: Motivo_Selecionado
    - **Coluna 7**: Comentario
    - **Coluna 9** (opcional): Grupo_Motivo
    
    #### 🌡️ Novos Recursos:
    - **Termômetro Emocional**: Classifica clientes em Feliz, Neutro, Atritado ou Extremamente Insatisfeito
    - **Identificação de OS Crítica**: Mostra a Ordem de Serviço do pior comentário para ação imediata
    - **Dashboard Visual**: Gráficos interativos para análise rápida
    
    ### 📋 Requirements.txt
    ```txt
    streamlit>=1.28.0
    pandas>=1.5.0
    numpy>=1.24.0
    openai>=1.0.0
    scikit-learn>=1.3.0
    openpyxl>=3.1.0
    plotly>=5.0.0
    ```
    """)
