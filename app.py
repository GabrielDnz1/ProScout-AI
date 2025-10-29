import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mplsoccer import PyPizza, FontManager
# --- Importações de Scikit-learn ---
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics.pairwise import cosine_similarity 
# -----------------------------------

st.set_page_config(layout="wide")
st.title("PROScout AI")

uploaded_file = st.file_uploader("📂 Carregue um arquivo CSV ou XLSX", type=["csv", "xlsx"])
page = st.sidebar.radio("Selecione a Ferramenta AI", ["Análise de Estilos", "Jogador Similar"]) 

if uploaded_file is not None:

    # -------------------------------
    # Ler CSV ou XLSX
    # -------------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = df.loc[:, ~df.columns.duplicated()]

    # Converter números com vírgula -> ponto
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                # Tenta converter colunas numéricas que usam vírgula como decimal
                df[col] = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float)
            except:
                pass 

    # -------------------------------
    # Filtros Comuns
    # -------------------------------
    col1_idade, col2_min = st.columns(2)
    
    # Valores default caso as colunas não existam
    idade_sel = (0, 100)
    minutesplayed_sel = (0, 99999)

    with col1_idade:
        if "Idade" in df.columns and df["Idade"].dtype in ['int64', 'float64']:
            idade_min, idade_max = int(df["Idade"].min()), int(df["Idade"].max())
            idade_sel = st.slider("Idade do jogador", idade_min, idade_max, (idade_min, idade_max))
        else:
            st.warning("Coluna 'Idade' não encontrada ou não é numérica. Filtro desativado.")

    with col2_min:
        if "Minutos jogados:" in df.columns and df["Minutos jogados:"].dtype in ['int64', 'float64']:
            minplayed_min, minplayed = int(df["Minutos jogados:"].min()), int(df["Minutos jogados:"].max())
            minutesplayed_sel = st.slider("Minutos do jogador na temporada", minplayed_min, minplayed, (minplayed_min, minplayed))
        else:
            st.warning("Coluna 'Minutos jogados:' não encontrada ou não é numérica. Filtro desativado.")
    
    # Aplica os filtros na base de dados
    df_temp = df.copy()
    if "Idade" in df_temp.columns:
        df_temp = df_temp[(df_temp["Idade"] >= idade_sel[0]) & (df_temp["Idade"] <= idade_sel[1])]
    if "Minutos jogados:" in df_temp.columns:
        df_filtrado_min_total = df_temp[(df_temp["Minutos jogados:"] >= minutesplayed_sel[0]) & (df_temp["Minutos jogados:"] <= minutesplayed_sel[1])].copy()
    else:
        df_filtrado_min_total = df_temp.copy()
    
    # -------------------------------
    # Mapeamentos de Estilos, Métricas e Pesos (DEFINIÇÕES)
    # -------------------------------
    posicoes_fixas = ["Goleiro", "Lateral", "Zagueiro", "Volante", 
                     "Meia-Central", "Meia-Ofensivo", "Extremo", "Centroavante"]
    
    estilos_pos = {
        "Centroavante": ["Finalizador", "Pressionador", "Dominador Aéreo", "Movimentador", "Assistente"],
        "Extremo": ["Driblador", "Finalizador", "Cruzador", "Acelerador", "Assistente"],
        "Meia-Ofensivo": ["Assistente", "Construtor", "Driblador", "Finalizador", "Especialista em Bola Parada"],
        "Meia-Central": ["Construtor", "Assistente", "Box-to-Box", "Recuperador", "Distribuidor"],
        "Volante": ["Recuperador", "Construtor", "Defensor", "Distribuidor", "Pressionador"],
        "Lateral": ["Construtor", "Cruzador", "Acelerador", "Desarme", "Movimentador"],
        "Zagueiro": ["Defensor", "Dominador Aéreo", "Construtor", "Líder de Defesa", "Lançador"],
        "Goleiro": ["Shot Stopper", "Sweeper Keeper", "Distribuidor"]
    }

    metricas_por_estilo = {
        "Shot Stopper": ["Defesas, %", "Golos sofridos/90", "Golos expectáveis defendidos por 90´"],
        "Sweeper Keeper": ["Saídas/90", "Duelos aéreos/90", "Duelos aéreos ganhos, %"],
        "Distribuidor": ["Passes certos, %", "Passes longos certos, %", "Passes para trás recebidos pelo guarda-redes/90"],
        "Defensor": ["Duelos defensivos/90", "Duelos defensivos ganhos, %", "Cortes/90", "Interseções/90", "Faltas/90"],
        "Líder de Defesa": ["Ações defensivas com êxito/90", "Duelos aéreos ganhos, %"],
        "Construtor": ["Passes/90", "Passes certos, %", "Passes progressivos/90", "Passes progressivos certos, %"],
        "Lançador": ["Passes longos/90", "Passes longos certos, %", "Passes em profundidade/90", "Passes em profundidade certos, %"],
        "Dominador Aéreo": ["Duelos aéreos/90", "Duelos aéreos ganhos, %", "Golos de cabeça/90"],
        "Cruzador": ["Cruzamentos/90", "Cruzamentos certos, %", "Passes para a área de penálti/90"],
        "Driblador": ["Dribles/90", "Dribles com sucesso, %", "Acelerações/90"],
        "Desarme": ["Duelos defensivos/90", "Duelos defensivos ganhos, %", "Interseções/90"],
        "Recuperador": ["Interseções/90", "Duelos defensivos/90", "Duelos defensivos ganhos, %", "Faltas/90"],
        "Box-to-Box": ["Duelos/90", "Interseções/90", "Corridas progressivas/90", "Acelerações/90"],
        "Assistente": ["Assistências/90", "Assistências esperadas/90", "Passes chave/90", "Passes inteligentes/90", "Passes inteligentes certos, %"],
        "Finalizador": ["Golos/90", "Remates/90", "Remates à baliza, %", "Golos esperados/90", "Toques na área/90"],
        "Acelerador": ["Corridas progressivas/90", "Acelerações/90"],
        "Pressionador": ["Duelos defensivos/90", "Duelos defensivos ganhos, %", "Acções atacantes com sucesso/90"],
        "Movimentador": ["Acelerações/90", "Corridas progressivas/90", "Passes recebidos/90"],
        "Especialista em Bola Parada": ["Assistências por bola parada/90", "Passes chave por bola parada/90"],
    }
    
    # Dicionário de Pesos para o Score (Ponderação) - Usado na Análise de Estilos
    pesos_por_estilo = {
        "Construtor": {"Passes certos, %": 3.0, "Passes progressivos certos, %": 2.5, "Passes progressivos/90": 1.5, "Passes/90": 1.0,},
        "Assistente": {"Assistências/90": 3.0, "Passes chave/90": 2.5, "Assistências esperadas/90": 2.0, "Passes inteligentes certos, %": 1.5,},
        "Driblador": {"Dribles com sucesso, %": 2.5, "Dribles/90": 1.5, "Acelerações/90": 1.0,},
        "Finalizador": {"Golos/90": 3.0, "Golos esperados/90": 2.5, "Remates à baliza, %": 1.5, "Remates/90": 1.0,},
        "Defensor": {"Duelos defensivos ganhos, %": 3.0, "Interseções/90": 2.5, "Cortes/90": 2.0, "Duelos defensivos/90": 1.0, "Faltas/90": 1.0,},
        "Líder de Defesa": {"Duelos aéreos ganhos, %": 3.0, "Ações defensivas com êxito/90": 2.0,},
        "Lançador": {"Passes longos certos, %": 3.0, "Passes em profundidade certos, %": 2.5, "Passes longos/90": 1.5, "Passes em profundidade/90": 1.0,},
        "Cruzador": {"Cruzamentos certos, %": 3.0, "Passes para a área de penálti/90": 2.0, "Cruzamentos/90": 1.0,},
        "Desarme": {"Duelos defensivos ganhos, %": 3.0, "Interseções/90": 2.0, "Duelos defensivos/90": 1.5,},
        "Recuperador": {"Duelos defensivos ganhos, %": 3.0, "Interseções/90": 2.5, "Duelos defensivos/90": 1.5, "Faltas/90": 1.0,},
        "Box-to-Box": {"Corridas progressivas/90": 2.5, "Interseções/90": 2.0, "Duelos/90": 1.5, "Acelerações/90": 1.0,},
        "Distribuidor": {"Passes certos, %": 3.0, "Passes curtos / médios precisos, %": 2.5, "Passes curtos / médios /90": 1.5,},
        "Acelerador": {"Corridas progressivas/90": 2.5, "Acelerações/90": 1.5,},
        "Pressionador": {"Duelos defensivos ganhos, %": 3.0, "Acções atacantes com sucesso/90": 2.0, "Duelos defensivos/90": 1.0,},
        "Dominador Aéreo": {"Golos de cabeça/90": 3.0, "Duelos aéreos ganhos, %": 2.0, "Duelos aéreos/90": 1.0,},
        "Movimentador": {"Passes recebidos/90": 2.5, "Corridas progressivas/90": 1.5, "Acelerações/90": 1.0,},
        "Shot Stopper": {"Defesas, %": 3.0, "Golos expectáveis defendidos por 90´": 2.5, "Golos sofridos/90": 1.0,},
        "Sweeper Keeper": {"Saídas/90": 2.0, "Duelos aéreos ganhos, %": 3.0, "Duelos aéreos/90": 1.0,},
    }
    
    # KPIs fixos para o radar por posição (e para similaridade)
    kpis_por_posicao = {
        "Goleiro": {
            "Defendendo": ["Defesas, %", "Golos sofridos/90", "Golos sofridos esperados/90", "Golos expectáveis defendidos por 90´", "Remates sofridos/90", "Jogos sem sofrer golos"],
            "Posse": ["Passes certos, %", "Passes longos/90", "Passes longos certos, %", "Passes para trás recebidos pelo guarda-redes/90", "Saídas/90"],
            "Atacando": []
        },
        "Zagueiro": {
            "Defendendo": ["Ações defensivas com êxito/90", "Duelos defensivos/90", "Duelos defensivos ganhos, %", "Cortes/90", "Cortes de carrinho ajust. à posse", "Remates intercetados/90", "Interseções/90", "Interceções ajust. à posse", "Duelos aéreos/90", "Duelos aéreos ganhos, %"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes para a frente/90", "Passes para a frente certos, %", "Passes laterais/90", "Passes laterais certos, %", "Passes progressivos/90", "Passes progressivos certos, %"],
            "Atacando": ["Golos", "Golos de cabeça/90", "Assistências/90"]
        },
        "Lateral": {
            "Defendendo": ["Duelos defensivos/90", "Duelos defensivos ganhos, %", "Interseções/90", "Cortes/90"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes progressivos/90", "Passes progressivos certos, %", "Corridas progressivas/90"],
            "Atacando": ["Assistências/90", "Assistências esperadas/90", "Cruzamentos/90", "Cruzamentos certos, %", "Cruzamentos do flanco esquerdo/90", "Cruzamentos precisos do flanco esquerdo, %", "Cruzamentos do flanco direito/90", "Cruzamentos precisos do flanco direito, %", "Acelerações/90"]
        },
        "Volante": {
            "Defendendo": ["Duelos defensivos/90", "Duelos defensivos ganhos, %", "Interseções/90", "Faltas/90"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes curtos / médios /90", "Passes curtos / médios precisos, %", "Passes para a frente/90", "Passes para a frente certos, %", "Passes progressivos/90", "Passes progressivos certos, %"],
            "Atacando": ["Assistências/90", "Assistências esperadas/90", "Passes chave/90", "Passes inteligentes/90"]
        },
        "Meia-Ofensivo": {
            "Defendendo": ["Duelos/90", "Duelos ganhos, %"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes chave/90", "Passes para terço final/90", "Passes certos para terço final, %", "Passes para a área de penálti/90", "Passes precisos para a área de penálti, %", "Passes inteligentes/90"],
            "Atacando": ["Golos/90", "Golos esperados/90", "Assistências/90", "Assistências esperadas/90", "Dribles/90", "Dribles com sucesso, %", "Toques na área/90"]
        },
        "Extremo": {
            "Defendendo": ["Duelos ofensivos/90", "Duelos ofensivos ganhos, %"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes progressivos/90", "Passes progressivos certos, %", "Corridas progressivas/90", "Acelerações/90"],
            "Atacando": ["Golos/90", "Golos esperados/90", "Assistências/90", "Assistências esperadas/90", "Cruzamentos/90", "Cruzamentos certos, %", "Dribles/90", "Dribles com sucesso, %", "Toques na área/90"]
        },
        "Centroavante": {
            "Defendendo": ["Ações defensivas com êxito/90", "Duelos aéreos/90", "Duelos aéreos ganhos, %"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes recebidos/90", "Passes longos recebidos/90"],
            "Atacando": ["Golos/90", "Golos sem ser por penálti/90", "Golos esperados/90", "Golos de cabeça/90", "Remates/90", "Remates à baliza, %", "Toques na área/90", "Acelerações/90"]
        }
    }


    # =======================================================
    # PÁGINA 1: ANÁLISE DE ESTILOS (PROSCOUT AI)
    # =======================================================
    if page == "Análise de Estilos":
        st.header("Análise de Estilos de Jogadores (Score Ponderado)")

        posicao_sel = st.selectbox("Selecione a posição (apenas para o gráfico)", posicoes_fixas)

        estilos_validos = estilos_pos.get(posicao_sel, [])
        estilos_escolhidos = st.multiselect("Selecione os estilos", estilos_validos)

        # -------------------------------
        # Botão gerar análise
        # -------------------------------
        if st.button("Gerar análise"):

            if df_filtrado_min_total.empty:
                st.warning("Nenhum jogador encontrado com esses filtros.")
            elif not estilos_escolhidos:
                st.warning("Selecione pelo menos um estilo para análise.")
            else:
                # Métricas selecionadas pelos estilos
                metricas = []
                for estilo in estilos_escolhidos:
                    metricas.extend(metricas_por_estilo.get(estilo, []))
                metricas_existentes = list(set([m for m in metricas if m in df_filtrado_min_total.columns])) # Usa set para remover duplicatas
                
                df_pos = df_filtrado_min_total.copy()

                if not metricas_existentes:
                    st.warning("Nenhuma métrica válida encontrada no dataset para os estilos selecionados.")
                else:
                    # Métrica para ranqueamento invertido
                    metricas_negativas = ["Golos sofridos/90", "Faltas/90"] 
                    
                    # Gerar percentuais para métricas dos estilos
                    for col in metricas_existentes:
                        if col in metricas_negativas:
                            df_pos[col + "_pct"] = df_pos[col].rank(pct=True, ascending=False) * 100
                        else:
                            df_pos[col + "_pct"] = df_pos[col].rank(pct=True) * 100

                    # ---------------------------------------------
                    # CÁLCULO DE SCORE COM PESOS (Ponderação)
                    # ---------------------------------------------
                    
                    pesos_finais = {}
                    for estilo in estilos_escolhidos:
                        pesos_estilo = pesos_por_estilo.get(estilo, {})
                        
                        # Tenta coletar pesos de estilos genéricos
                        if not pesos_estilo:
                            pesos_estilo = pesos_por_estilo.get(estilo, {})
                        
                        # Adiciona os pesos das métricas
                        for metrica, peso in pesos_estilo.items():
                            if metrica in metricas_existentes:
                                # Usa o maior peso se a métrica for relevante para múltiplos estilos
                                pesos_finais[metrica] = max(pesos_finais.get(metrica, 0.0), peso)
                            
                    df_pos["Score Ponderado"] = 0.0
                    soma_pesos = sum(pesos_finais.values())
                    
                    if soma_pesos > 0:
                        for metrica, peso in pesos_finais.items():
                            df_pos["Score Ponderado"] += df_pos[metrica + "_pct"] * peso
                            
                        df_pos["Score"] = df_pos["Score Ponderado"] / soma_pesos
                    else:
                        # Fallback para média simples (original) se não houver pesos definidos
                        df_pos["Score"] = df_pos[[c+"_pct" for c in metricas_existentes]].mean(axis=1)

                    df_final = df_pos.sort_values(by="Score", ascending=False)
                    
                    st.dataframe(df_final[["Jogador", "Equipa", "Idade", "Score"] + metricas_existentes].round(1))


                    # -------------------------------
                    # Radar do melhor jogador
                    # -------------------------------
                    
                    # 1. Lista de todas as métricas que podem ir no radar (para calcular os percentis)
                    todas_metricas_radar = []
                    for kpis_pos in kpis_por_posicao.values():
                        for grupo_metrica in kpis_pos.values():
                            todas_metricas_radar.extend(grupo_metrica)
                    todas_metricas_radar = list(set([m for m in todas_metricas_radar if m in df_final.columns]))
                    
                    # 2. Aplicar ranqueamento (percentil) para o radar
                    metricas_negativas = ["Golos sofridos/90", "Faltas/90"] 
                    for col in todas_metricas_radar:
                        if col + "_pct" not in df_final.columns: # Não recalcular o que já existe
                             if col in metricas_negativas:
                                 df_final[col + "_pct"] = df_final[col].rank(pct=True, ascending=False) * 100
                             else:
                                 df_final[col + "_pct"] = df_final[col].rank(pct=True) * 100

                    if df_final.empty:
                        st.warning("Não há jogadores para plotar no radar.")
                    else:
                        top_player = df_final.iloc[0]
                        st.subheader(f"Jogador Sugerido - {top_player.get('Jogador', 'N/A')} ({posicao_sel})")

                        kpis = kpis_por_posicao.get(posicao_sel, {})
                        metricas_ordenadas = []
                        valores = []
                        slice_colors = []
                        grupo_cores = {"Atacando": "#FF5733", "Defendendo": "#33FF57", "Posse": "#3375FF"}

                        for grupo, metricas in kpis.items():
                            for metrica in metricas:
                                pct_col = metrica + "_pct"
                                if pct_col in df_final.columns:
                                    metricas_ordenadas.append(metrica)
                                    valor = df_final.loc[df_final["Jogador"] == top_player["Jogador"], pct_col].values[0]
                                    valores.append(round(float(valor), 2))
                                    slice_colors.append(grupo_cores.get(grupo, "#999999"))

                        if metricas_ordenadas:
                            try:
                                font_normal = FontManager("https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf")
                                font_bold = FontManager("https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab[wght].ttf")

                                baker = PyPizza(
                                    params=metricas_ordenadas,
                                    background_color="#ffffff",
                                    straight_line_color="#cccccc",
                                    straight_line_lw=1,
                                    last_circle_lw=0,
                                    other_circle_lw=0,
                                    inner_circle_size=20
                                )

                                fig, ax = baker.make_pizza(
                                    valores,
                                    figsize=(6, 6),
                                    color_blank_space="same",
                                    slice_colors=slice_colors,
                                    value_colors=["#000000"] * len(valores),
                                    kwargs_slices=dict(edgecolor="#ffffff", zorder=2, linewidth=1),
                                    kwargs_params=dict(color="#000000", fontsize=4, fontproperties=font_normal.prop, va="center"),
                                    kwargs_values=dict(color="#000000", fontsize=8, fontproperties=font_bold.prop, zorder=3, va="center")
                                )

                                fig.text(
                                    0.5, 0.97,
                                    f"{top_player.get('Jogador', 'N/A')} - {top_player.get('Equipa', 'N/A')} ({', '.join(estilos_escolhidos)})",
                                    size=12, ha="center", fontproperties=font_bold.prop, color="#000000"
                                )

                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Erro ao gerar o gráfico de radar: {e}")
                        else:
                            st.warning("Não há métricas disponíveis para o radar desta posição.")

# =======================================================
    # PÁGINA 2: PROSCOUT AI (JOGADOR SIMILAR) - UNIVERSAL E ROBUSTA
    # =======================================================
    if page == "Jogador Similar":
        st.header("🔍 Encontre Jogadores Similares (AI Similarity - Busca Universal Segmentada)")
        
        # --- 0. Preparação e Criação de Chave Única ---
        chave_unica_disponivel = False
        options = ['-- Colunas ' + ', '.join(['Jogador', 'Equipa', 'Posição']) + ' não encontradas --']
        jogador_referencia = None
        
        if 'Jogador' in df.columns and 'Equipa' in df.columns and 'Posição' in df.columns:
            df_calculo = df_filtrado_min_total.copy()
            df_calculo['Chave_Unica'] = df_calculo['Jogador'] + " (" + df_calculo['Equipa'] + ")"
            
            if not df_calculo.empty:
                chave_unica_disponivel = True
                options = df_calculo['Chave_Unica'].unique().tolist()
                
                if not options:
                    options = ['-- Nenhum jogador elegível --']

        # O seletor AGORA É SEMPRE EXIBIDO
        jogador_referencia_chave = st.selectbox("1. Selecione o Jogador de Referência (Nome + Equipa):", options)

        ref_player_data_row = None # Inicializa para guardar dados do jogador ref
        posicao_contexto = None
        tipo_jogador = None

        if chave_unica_disponivel and jogador_referencia_chave != '-- Nenhum jogador elegível --' and jogador_referencia_chave in df_calculo['Chave_Unica'].values:
            ref_player_data_row = df_calculo[df_calculo['Chave_Unica'] == jogador_referencia_chave]
            if not ref_player_data_row.empty:
                jogador_referencia = ref_player_data_row['Jogador'].iloc[0]
                posicao_contexto = ref_player_data_row['Posição'].iloc[0]
                
                # Detecção do TIPO de jogador (Goleiro vs Linha)
                tipo_jogador = 'Goleiro' if posicao_contexto == 'Goleiro' else 'Linha'
                
                st.info(f"O jogador de referência '{jogador_referencia}' joga como: **{posicao_contexto}** (A busca será segmentada por **{tipo_jogador}**).")
            else:
                 jogador_referencia = None
                 st.warning("Jogador de referência não encontrado no conjunto de dados filtrado. Tente ajustar os filtros.")
        
        # REMOVIDO: O seletor de Mínimo de Minutos (item 2)
        
        if st.button("Buscar Jogadores Similares") and jogador_referencia is not None:
            
            if not chave_unica_disponivel or ref_player_data_row is None:
                st.error("Não é possível executar a busca. Verifique se as colunas estão corretas e se o jogador selecionado é válido.")
            else:
                # --- 1. Definir Métricas Segmentadas ---
                if tipo_jogador == 'Goleiro':
                        # Usa APENAS métricas de goleiro
                        metricas_sim = kpis_por_posicao.get('Goleiro', {}).get('Defendendo', []) + \
                                       kpis_por_posicao.get('Goleiro', {}).get('Posse', [])
                else:
                        # Combina métricas de TODAS as posições de linha
                        metricas_sim = []
                        for pos, kpis in kpis_por_posicao.items():
                            if pos != 'Goleiro':
                                for grupo_metrica in kpis.values():
                                    metricas_sim.extend(grupo_metrica)
                
                metricas_sim = list(set([m for m in metricas_sim if m in df.columns]))
    
                if not metricas_sim:
                    st.warning("Nenhuma métrica de comparação válida encontrada para o tipo de jogador. Verifique as colunas.")
                    can_proceed = False
                else:
                    can_proceed = True
    
    
                if can_proceed:
                    
                    # --- 2. Filtrar Pool de Busca (Segmentado por Tipo) ---
                    pool_busca = df_calculo.copy() # DataFrame com Chave_Unica
                    pool_busca = pool_busca[pool_busca['Chave_Unica'] != jogador_referencia_chave] # Remove o próprio
                    
                    # Filtra o pool de busca pelo tipo de jogador (Goleiro ou Linha)
                    if tipo_jogador == 'Goleiro':
                        pool_busca = pool_busca[pool_busca['Posição'] == 'Goleiro'].copy()
                    else:
                        pool_busca = pool_busca[pool_busca['Posição'] != 'Goleiro'].copy()
                    
                    # Extrair o dado de referência (já temos em ref_player_data_row)
                    
                    if ref_player_data_row.empty:
                        st.error(f"Erro: Jogador '{jogador_referencia_chave}' não encontrado no pool de dados.")
                        can_proceed = False
                    
                    if pool_busca.empty:
                        st.warning(f"Nenhum outro jogador do tipo **{tipo_jogador}** encontrado no pool de busca para comparação.")
                        can_proceed = False
                    
                    if can_proceed:
                        # 3. Preparar os dados (usando Chave_Unica como índice)
                        df_sim = pool_busca[['Chave_Unica'] + metricas_sim].set_index('Chave_Unica').fillna(0)
                        ref_data = ref_player_data_row[metricas_sim].fillna(0).iloc[0].to_frame().T
                        
                        scaler_sim = StandardScaler()
                        
                        if len(df_sim) > 1:
                            # Escalonamento se houver dados suficientes no pool
                            df_sim_scaled = scaler_sim.fit_transform(df_sim)
                            df_sim_scaled = pd.DataFrame(df_sim_scaled, columns=metricas_sim, index=df_sim.index)
                            ref_vector_scaled = scaler_sim.transform(ref_data).reshape(1, -1)
                        else:
                            st.info("Pool de busca pequeno. O cálculo será feito sem normalização.")
                            df_sim_scaled = df_sim
                            ref_vector_scaled = ref_data.values.reshape(1, -1)
                            
    
                        # 4. Calcular Similaridade (Cosine Similarity)
                        similarity_scores = cosine_similarity(ref_vector_scaled, df_sim_scaled)
                        
                        # 5. Criar DataFrame de Resultados (indexado pela Chave Única)
                        df_results = pd.DataFrame(similarity_scores.T, index=df_sim_scaled.index, columns=['Similaridade'])
                        
                        # CONVERTE SIMILARIDADE DE [-1, 1] (cosine) PARA [0, 100]
                        # Um scaler simples (MinMaxScaler) é mais robusto que multiplicar por 100
                        scaler_display = MinMaxScaler(feature_range=(0, 100))
                        
                        # Se todos os scores forem iguais (ex: 0 ou 1), o fit falha.
                        if (df_results['Similaridade'].max() - df_results['Similaridade'].min()) > 0:
                            df_results['Similaridade'] = scaler_display.fit_transform(df_results[['Similaridade']])
                        else:
                             # Se todos os valores são iguais, podemos apenas setar para 100 ou 0
                             df_results['Similaridade'] = 100.0 if df_results['Similaridade'].iloc[0] > 0 else 0.0

                        
                        df_results = df_results.sort_values(by='Similaridade', ascending=False)
                        
                        # Obtém as CHAVES ÚNICAS dos top 5
                        top_similares_chaves = df_results.head(5).index.tolist()
                        
                        # 6. Exibir Resultados
                        st.subheader(f"Top 5 Jogadores Mais Similares a: **{jogador_referencia}** (Busca {tipo_jogador})")
                        
                        # Pega as linhas dos jogadores similares
                        df_display = df_calculo[df_calculo['Chave_Unica'].isin(top_similares_chaves)].set_index('Chave_Unica')
                        
                        # Reordena e mescla a pontuação de similaridade
                        df_display = df_display.reindex(top_similares_chaves)
                        df_display = df_display.join(df_results, how='left')
                        
                        # Colunas relevantes para o display final
                        display_cols = ['Jogador', 'Equipa', 'Idade', 'Posição', 'Similaridade', 'Minutos jogados:']
                        
                        # Filtra colunas que realmente existem no df_display
                        df_display = df_display[[col for col in display_cols if col in df_display.columns]].round(2)
                        
                        # Exibe a similaridade como porcentagem de 0 a 100 com barra de progresso
                        st.dataframe(df_display, 
                                     column_config={"Similaridade": st.column_config.ProgressColumn(
                                         "Similaridade (%)", 
                                         format="%.2f %%", 
                                         min_value=0, 
                                         max_value=100
                                     )})

                        # ---------------------------------------------------
                        # (NOVO) 7. GRÁFICO RADAR COMPARATIVO
                        # ---------------------------------------------------
                        
                        if not top_similares_chaves:
                            st.info("Nenhum jogador similar encontrado para comparar no radar.")
                            return # Sai se a lista estiver vazia

                        st.subheader(f"Comparativo Visual: {jogador_referencia} vs. {df_display.iloc[0]['Jogador']}")

                        # 7.1 Obter dados do jogador Top 1 similar
                        top_similar_data_row = df_calculo[df_calculo['Chave_Unica'] == top_similares_chaves[0]]

                        # 7.2 Posição (usar a do jogador de referência como base)
                        posicao_radar = posicao_contexto # Já definido acima
                        kpis = kpis_por_posicao.get(posicao_radar, {})

                        # 7.3 Calcular Percentis para o Radar
                        # Precisamos usar o 'df_calculo' que contém todos os jogadores filtrados (idade, minutos)
                        
                        todas_metricas_radar = []
                        for grupo_metrica in kpis.values():
                            todas_metricas_radar.extend(grupo_metrica)
                        
                        metricas_radar_existentes = list(set([m for m in todas_metricas_radar if m in df_calculo.columns]))
                        
                        # Usamos df_calculo para o rank, pois ele tem o pool completo de jogadores
                        df_radar_pct = df_calculo.copy() 
                        metricas_negativas = ["Golos sofridos/90", "Faltas/90"] # Reutilizar
                        
                        for col in metricas_radar_existentes:
                            pct_col_name = col + "_pct"
                            if col in metricas_negativas:
                                df_radar_pct[pct_col_name] = df_radar_pct[col].rank(pct=True, ascending=False) * 100
                            else:
                                df_radar_pct[pct_col_name] = df_radar_pct[col].rank(pct=True) * 100

                        # 7.4 Obter os valores de percentil para os DOIS jogadores
                        valores_ref = []
                        valores_sim = []
                        metricas_ordenadas = []
                        slice_colors = []
                        grupo_cores = {"Atacando": "#FF5733", "Defendendo": "#33FF57", "Posse": "#3375FF"} # Reutilizar

                        # Dados de percentil do jogador de referência
                        player_ref_pct = df_radar_pct[df_radar_pct['Chave_Unica'] == jogador_referencia_chave]
                        # Dados de percentil do jogador similar
                        player_sim_pct = df_radar_pct[df_radar_pct['Chave_Unica'] == top_similares_chaves[0]]

                        if player_ref_pct.empty or player_sim_pct.empty:
                            st.warning("Não foi possível gerar o radar comparativo (dados de percentil não encontrados).")
                        else:
                            for grupo, metricas in kpis.items():
                                for metrica in metricas:
                                    pct_col = metrica + "_pct"
                                    if pct_col in df_radar_pct.columns:
                                        metricas_ordenadas.append(metrica)
                                        # Valor do jogador de referência
                                        valor_ref = player_ref_pct[pct_col].values[0]
                                        valores_ref.append(round(float(valor_ref), 2))
                                        # Valor do jogador similar
                                        valor_sim = player_sim_pct[pct_col].values[0]
                                        valores_sim.append(round(float(valor_sim), 2))
                                        
                                        slice_colors.append(grupo_cores.get(grupo, "#999999"))

                            # 7.5 Plotar o Gráfico (usando PyPizza)
                            if metricas_ordenadas:
                                try:
                                    font_normal = FontManager("https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf")
                                    font_bold = FontManager("https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab[wght].ttf")

                                    baker = PyPizza(
                                        params=metricas_ordenadas,
                                        background_color="#ffffff",
                                        straight_line_color="#cccccc",
                                        straight_line_lw=1,
                                        last_circle_lw=0,
                                        other_circle_lw=0,
                                        inner_circle_size=20
                                    )

                                    fig, ax = baker.make_pizza(
                                        valores_ref,              # Valores do Jogador 1 (Referência)
                                        compare_values=valores_sim, # Valores do Jogador 2 (Similar)
                                        figsize=(6, 6),
                                        color_blank_space="same",
                                        slice_colors=slice_colors, # Cor principal (Referência)
                                        value_colors=["#000000"] * len(valores_ref),
                                        
                                        # Estilo Jogador 1 (Referência)
                                        kwargs_slices=dict(edgecolor="#ffffff", zorder=2, linewidth=1),
                                        kwargs_params=dict(color="#000000", fontsize=4, fontproperties=font_normal.prop, va="center"),
                                        kwargs_values=dict(color="#000000", fontsize=8, fontproperties=font_bold.prop, zorder=3, va="center"),
                                        
                                        # Estilo Jogador 2 (Similar)
                                        compare_color="#800080", # Roxo para o similar
                                        kwargs_compare=dict(
                                            facecolor="#800080", edgecolor="#ffffff", zorder=2, linewidth=1, alpha=0.6 # Com transparência
                                        ),
                                        kwargs_compare_values=dict(
                                            color="#800080", fontsize=8, fontproperties=font_bold.prop, zorder=3, va="center", alpha=0.7
                                        )
                                    )

                                    # Título e Legenda
                                    ref_nome = ref_player_data_row['Jogador'].iloc[0]
                                    sim_nome = top_similar_data_row['Jogador'].iloc[0]
                                    
                                    fig.text(
                                        0.5, 0.97,
                                        f"{ref_nome} (Ref) vs. {sim_nome} (Similar)",
                                        size=12, ha="center", fontproperties=font_bold.prop, color="#000000"
                                    )
                                    
                                    # Legenda customizada
                                    fig.text(
                                        0.35, 0.92,
                                        f"■ {ref_nome}",
                                        size=10, ha="center", fontproperties=font_bold.prop, color=slice_colors[0] # Uma cor base
                                    )
                                    fig.text(
                                        0.65, 0.92,
                                        f"■ {sim_nome}",
                                        size=10, ha="center", fontproperties=font_bold.prop, color="#800080" # Cor Roxo
                                    )

                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Erro ao gerar o gráfico de radar comparativo: {e}")
                            else:
                                st.warning(f"Não há métricas de radar disponíveis para a posição: {posicao_radar}")
