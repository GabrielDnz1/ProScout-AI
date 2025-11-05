import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mplsoccer import PyPizza, FontManager
# --- Importações para a Análise de Similaridade ---
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics.pairwise import cosine_similarity 
# --------------------------------------------------

st.set_page_config(layout="wide")
st.title("PROScout AI")

# --- Link de Download para a Base Modelo ---
# Oferece ao usuário um link direto para baixar um arquivo de exemplo, útil para testar a aplicação.
st.markdown("🔗 **Baixe o arquivo modelo - Todos os Jogadores do brasileirão com mais de 500 minutos (Wyscout CSV):** [Modelo de Base de Dados](https://drive.google.com/file/d/1ohP0Jfv0Sx3C5ILwvSXuOTXEUGGiMHNq/view?usp=sharing)")
# ---------------------------------------------

# Permite ao usuário carregar a própria base de dados
uploaded_file = st.file_uploader("📂 Carregue um arquivo CSV ou XLSX", type=["csv", "xlsx"])
# Menu lateral para alternar entre as funcionalidades da AI
page = st.sidebar.radio("Selecione a Ferramenta AI", ["Análise de Estilos", "Jogador Similar"]) 

if uploaded_file is not None:

    # -------------------------------
    # Carregamento e Limpeza Inicial dos Dados
    # -------------------------------
    # Identifica o tipo de arquivo (.csv ou .xlsx) e carrega para um DataFrame do pandas.
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Remove colunas duplicadas que podem surgir de bases de dados mal formatadas.
    df = df.loc[:, ~df.columns.duplicated()]

    # Trata a formatação de números com vírgula: substitui ',' por '.' e converte para float.
    # Essencial para garantir que as métricas sejam reconhecidas como numéricas para os cálculos.
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                # Tenta converter colunas numéricas que usam vírgula como decimal
                df[col] = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float)
            except:
                pass 

    # -------------------------------
    # Interface e Aplicação de Filtros de Idade e Minutos
    # -------------------------------
    col1_idade, col2_min = st.columns(2)
    
    # Define valores padrão para os filtros caso as colunas necessárias não existam.
    idade_sel = (0, 100)
    minutesplayed_sel = (0, 99999)

    with col1_idade:
        # Exibe o slider de idade se a coluna estiver presente e for numérica.
        if "Idade" in df.columns and df["Idade"].dtype in ['int64', 'float64']:
            idade_min, idade_max = int(df["Idade"].min()), int(df["Idade"].max())
            idade_sel = st.slider("Idade do jogador", idade_min, idade_max, (idade_min, idade_max))
        else:
            st.warning("Coluna 'Idade' não encontrada ou não é numérica. Filtro desativado.")

    with col2_min:
        # Exibe o slider de minutos jogados se a coluna estiver presente e for numérica.
        if "Minutos jogados:" in df.columns and df["Minutos jogados:"].dtype in ['int64', 'float64']:
            minplayed_min, minplayed = int(df["Minutos jogados:"].min()), int(df["Minutos jogados:"].max())
            minutesplayed_sel = st.slider("Minutos do jogador na temporada", minplayed_min, minplayed, (minplayed_min, minplayed))
        else:
            st.warning("Coluna 'Minutos jogados:' não encontrada ou não é numérica. Filtro desativado.")
    
    # Cria uma cópia da base de dados e aplica os filtros de idade e minutos selecionados.
    df_temp = df.copy()
    if "Idade" in df_temp.columns:
        df_temp = df_temp[(df_temp["Idade"] >= idade_sel[0]) & (df_temp["Idade"] <= idade_sel[1])]
    if "Minutos jogados:" in df_temp.columns:
        df_filtrado_min_total = df_temp[(df_temp["Minutos jogados:"] >= minutesplayed_sel[0]) & (df_temp["Minutos jogados:"] <= minutesplayed_sel[1])].copy()
    else:
        df_filtrado_min_total = df_temp.copy()
    
    # -------------------------------
    # Dicionários de Configuração da AI (Estilos, Métricas e Pesos)
    # -------------------------------
    # Lista das principais posições para exibição no filtro.
    posicoes_fixas = ["Goleiro", "Lateral", "Zagueiro", "Volante", 
                      "Meia-Central", "Meia-Ofensivo", "Extremo", "Centroavante"]
    
    # Mapeamento de quais 'Estilos de Jogo' estão disponíveis para cada Posição.
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

    # Define as métricas (KPIs) necessárias para calcular a performance em cada 'Estilo de Jogo'.
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
    
    # Pesos definidos para cada métrica dentro de um Estilo. Usado para calcular o 'Score Ponderado'.
    # Um peso maior (ex: 3.0) indica uma métrica mais crítica para aquele estilo.
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
    
    # KPIs agrupados para a visualização no Gráfico de Radar (Pizza Plot) e para a Similaridade.
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

        # Seleção da posição, usada para definir os estilos disponíveis e o radar.
        posicao_sel = st.selectbox("Selecione a posição (apenas para o gráfico)", posicoes_fixas)

        # Seleção dos estilos, exibindo apenas os válidos para a posição escolhida.
        estilos_validos = estilos_pos.get(posicao_sel, [])
        estilos_escolhidos = st.multiselect("Selecione os estilos", estilos_validos)

        # -------------------------------
        # Botão para Iniciar o Cálculo e Análise
        # -------------------------------
        if st.button("Gerar análise"):

            if df_filtrado_min_total.empty:
                st.warning("Nenhum jogador encontrado com esses filtros.")
            elif not estilos_escolhidos:
                st.warning("Selecione pelo menos um estilo para análise.")
            else:
                # Coleta todas as métricas dos estilos escolhidos.
                metricas = []
                for estilo in estilos_escolhidos:
                    metricas.extend(metricas_por_estilo.get(estilo, []))
                # Filtra as métricas para incluir apenas as que existem na base de dados.
                metricas_existentes = list(set([m for m in metricas if m in df_filtrado_min_total.columns])) # Usa set para remover duplicatas
                
                df_pos = df_filtrado_min_total.copy()

                if not metricas_existentes:
                    st.warning("Nenhuma métrica válida encontrada no dataset para os estilos selecionados.")
                else:
                    # Lista de métricas em que um valor MENOR é melhor (para inverter o ranqueamento).
                    metricas_negativas = ["Golos sofridos/90", "Faltas/90"] 
                    
                    # Gera os percentis (rankings de 0 a 100) para cada métrica em relação aos outros jogadores.
                    for col in metricas_existentes:
                        if col in metricas_negativas:
                            # Inverte o ranqueamento (ex: menos Gols Sofridos = percentil mais alto)
                            df_pos[col + "_pct"] = df_pos[col].rank(pct=True, ascending=False) * 100
                        else:
                            # Ranqueamento normal (mais Dribles = percentil mais alto)
                            df_pos[col + "_pct"] = df_pos[col].rank(pct=True) * 100

                    # ---------------------------------------------
                    # CÁLCULO DE SCORE COM PESOS (Ponderação)
                    # ---------------------------------------------
                    
                    # Agrega os pesos de todas as métricas dos estilos selecionados.
                    pesos_finais = {}
                    for estilo in estilos_escolhidos:
                        pesos_estilo = pesos_por_estilo.get(estilo, {})
                        
                        if not pesos_estilo:
                            pesos_estilo = pesos_por_estilo.get(estilo, {})
                        
                        # Atribui o peso à métrica (mantendo o maior peso se for de múltiplos estilos)
                        for metrica, peso in pesos_estilo.items():
                            if metrica in metricas_existentes:
                                pesos_finais[metrica] = max(pesos_finais.get(metrica, 0.0), peso)
                            
                    df_pos["Score Ponderado"] = 0.0
                    soma_pesos = sum(pesos_finais.values())
                    
                    if soma_pesos > 0:
                        # Calcula o score: soma dos percentis ponderados pela importância (peso) da métrica.
                        for metrica, peso in pesos_finais.items():
                            df_pos["Score Ponderado"] += df_pos[metrica + "_pct"] * peso
                            
                        # Normaliza para um score final de 0 a 100.
                        df_pos["Score"] = df_pos["Score Ponderado"] / soma_pesos
                    else:
                        # Fallback: média simples se não houver pesos definidos.
                        df_pos["Score"] = df_pos[[c+"_pct" for c in metricas_existentes]].mean(axis=1)

                    # Classifica os jogadores pelo score final e exibe os resultados na tabela.
                    df_final = df_pos.sort_values(by="Score", ascending=False)
                    
                    st.dataframe(df_final[["Jogador", "Equipa", "Idade", "Score"] + metricas_existentes].round(1))


                    # -------------------------------
                    # Geração do Gráfico de Radar (Pizza Plot) para o Melhor Jogador
                    # -------------------------------
                    
                    # 1. Coleta todas as métricas necessárias para o radar de qualquer posição.
                    todas_metricas_radar = []
                    for kpis_pos in kpis_por_posicao.values():
                        for grupo_metrica in kpis_pos.values():
                            todas_metricas_radar.extend(grupo_metrica)
                    todas_metricas_radar = list(set([m for m in todas_metricas_radar if m in df_final.columns]))
                    
                    # 2. Garante que os percentis para o radar estejam calculados (incluindo as não usadas no score).
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
                        # Seleciona o jogador com o maior Score Ponderado
                        top_player = df_final.iloc[0]
                        st.subheader(f"Jogador Sugerido - {top_player.get('Jogador', 'N/A')} ({posicao_sel})")

                        # Prepara os dados do jogador (métricas e cores) para o radar.
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

                        # Cria e exibe o gráfico de radar PyPizza.
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
    # PÁGINA 2: ENCONTRAR JOGADOR SIMILAR
    # =======================================================
    if page == "Jogador Similar":
        st.header("🔍 Encontre Jogadores Similares (AI Similarity - Busca Universal Segmentada)")
        
        # --- 0. Preparação e Criação de Chave Única ---
        chave_unica_disponivel = False
        options = ['-- Colunas ' + ', '.join(['Jogador', 'Equipa', 'Posição']) + ' não encontradas --']
        jogador_referencia = None
        
        # Verifica se as colunas essenciais para a busca estão presentes.
        if 'Jogador' in df.columns and 'Equipa' in df.columns and 'Posição' in df.columns:
            df_calculo = df_filtrado_min_total.copy()
            # Cria a chave única "Jogador (Equipa)" para evitar nomes repetidos no seletor.
            df_calculo['Chave_Unica'] = df_calculo['Jogador'] + " (" + df_calculo['Equipa'] + ")"
            
            # Garante que cada jogador/equipe tenha uma única linha para evitar erros de índice.
            df_calculo = df_calculo.drop_duplicates(subset=['Chave_Unica'], keep='first')
            
            if not df_calculo.empty:
                chave_unica_disponivel = True
                options = df_calculo['Chave_Unica'].unique().tolist()
                
                if not options:
                    options = ['-- Nenhum jogador elegível --']

        # Seletor para escolher o jogador-modelo para a busca de similaridade.
        jogador_referencia_chave = st.selectbox("1. Selecione o Jogador de Referência (Nome + Equipa):", options)

        ref_player_data_row = None # Inicializa para guardar dados do jogador ref
        posicao_contexto = None
        tipo_jogador = None

        # Processa a seleção do jogador, define a posição e o "Tipo" (Goleiro ou Linha) para segmentar a busca.
        if chave_unica_disponivel and jogador_referencia_chave != '-- Nenhum jogador elegível --' and jogador_referencia_chave in df_calculo['Chave_Unica'].values:
            ref_player_data_row = df_calculo[df_calculo['Chave_Unica'] == jogador_referencia_chave]
            if not ref_player_data_row.empty:
                jogador_referencia = ref_player_data_row['Jogador'].iloc[0]
                posicao_contexto = ref_player_data_row['Posição'].iloc[0] # Posição bruta (ex: LCB)
                
                # A similaridade será comparada apenas entre Goleiros OU Jogadores de Linha.
                tipo_jogador = 'Goleiro' if posicao_contexto == 'Goleiro' else 'Linha' 
                
                st.info(f"O jogador de referência '{jogador_referencia}' joga como: **{posicao_contexto}** (A busca será segmentada por **{tipo_jogador}**).")
            else:
                jogador_referencia = None
                st.warning("Jogador de referência não encontrado no conjunto de dados filtrado. Tente ajustar os filtros.")
        
        
        if st.button("Buscar Jogadores Similares") and jogador_referencia is not None:
            
            if not chave_unica_disponivel or ref_player_data_row is None:
                st.error("Não é possível executar a busca. Verifique se as colunas estão corretas e se o jogador selecionado é válido.")
            else:
                # --- 1. Definir Métricas Segmentadas (Goleiro ou Linha) ---
                if tipo_jogador == 'Goleiro':
                    # Usa as métricas de defesa e posse de bola do goleiro.
                    metricas_sim = kpis_por_posicao.get('Goleiro', {}).get('Defendendo', []) + \
                                   kpis_por_posicao.get('Goleiro', {}).get('Posse', [])
                else:
                    # Combina métricas de TODAS as posições de linha para uma busca universal.
                    metricas_sim = []
                    for pos, kpis in kpis_por_posicao.items():
                        if pos != 'Goleiro':
                            for grupo_metrica in kpis.values():
                                metricas_sim.extend(grupo_metrica)
                
                # Filtra as métricas de comparação para incluir apenas as que estão na base.
                metricas_sim = list(set([m for m in metricas_sim if m in df.columns]))
        
                if not metricas_sim:
                    st.warning("Nenhuma métrica de comparação válida encontrada para o tipo de jogador. Verifique as colunas.")
                    can_proceed = False
                else:
                    can_proceed = True
        
                
                if can_proceed:
                    
                    # --- 2. Filtrar Pool de Busca (Segmentado por Tipo) ---
                    pool_busca = df_calculo.copy()
                    # Remove o próprio jogador de referência do pool de busca.
                    pool_busca = pool_busca[pool_busca['Chave_Unica'] != jogador_referencia_chave]
                    
                    # Filtra o pool para incluir apenas jogadores do mesmo tipo (Goleiro ou Linha).
                    if tipo_jogador == 'Goleiro':
                        pool_busca = pool_busca[pool_busca['Posição'] == 'Goleiro'].copy()
                    else:
                        pool_busca = pool_busca[pool_busca['Posição'] != 'Goleiro'].copy()
                    
                    if ref_player_data_row.empty:
                        st.error(f"Erro: Jogador '{jogador_referencia_chave}' não encontrado no pool de dados.")
                        can_proceed = False
                    
                    if pool_busca.empty:
                        st.warning(f"Nenhum outro jogador do tipo **{tipo_jogador}** encontrado no pool de busca para comparação.")
                        can_proceed = False
                    
                    if can_proceed:
                        # 3. Preparar os dados para cálculo.
                        # Normalização: Padroniza as métricas (média 0, desvio padrão 1) para que métricas com grandes valores não dominem a similaridade.
                        df_sim = pool_busca[['Chave_Unica'] + metricas_sim].set_index('Chave_Unica').fillna(0)
                        ref_data = ref_player_data_row[metricas_sim].fillna(0).iloc[0].to_frame().T
                        
                        scaler_sim = StandardScaler()
                        
                        if len(df_sim) > 1:
                            df_sim_scaled = scaler_sim.fit_transform(df_sim)
                            df_sim_scaled = pd.DataFrame(df_sim_scaled, columns=metricas_sim, index=df_sim.index)
                            # Transforma os dados do jogador de referência usando o mesmo scaler.
                            ref_vector_scaled = scaler_sim.transform(ref_data).reshape(1, -1)
                        else:
                            st.info("Pool de busca pequeno. O cálculo será feito sem normalização.")
                            df_sim_scaled = df_sim
                            ref_vector_scaled = ref_data.values.reshape(1, -1)
                            
                        # 4. Calcular Similaridade (Cosseno)
                        # A similaridade do cosseno mede o ângulo entre dois vetores de características.
                        similarity_scores = cosine_similarity(ref_vector_scaled, df_sim_scaled)
                        
                        # 5. Criar DataFrame de Resultados
                        df_results = pd.DataFrame(similarity_scores.T, index=df_sim_scaled.index, columns=['Similaridade'])
                        
                        # Converte o valor do cosseno (0 a 1) para porcentagem (0% a 100%).
                        df_results['Similaridade'] = df_results['Similaridade'] * 100
                        df_results['Similaridade'] = df_results['Similaridade'].clip(0, 100) # Limita por segurança

                        
                        df_results = df_results.sort_values(by='Similaridade', ascending=False)
                        
                        # 6. Exibir Resultados (Tabela)
                        top_similares_chaves = df_results.head(5).index.tolist()
                        st.subheader(f"Top 5 Jogadores Mais Similares a: **{jogador_referencia}** (Busca {tipo_jogador})")
                        
                        # Junta a similaridade com os dados originais do jogador.
                        df_display = df_calculo[df_calculo['Chave_Unica'].isin(top_similares_chaves)].set_index('Chave_Unica')
                        
                        # Reordena para exibir na ordem do mais similar para o menos.
                        df_display = df_display.reindex(top_similares_chaves)
                        
                        df_display = df_display.join(df_results, how='left')
                        
                        display_cols = ['Jogador', 'Equipa', 'Idade', 'Posição', 'Similaridade', 'Minutos jogados:']
                        df_display = df_display[[col for col in display_cols if col in df_display.columns]].round(2)
                        
                        # Exibe a tabela com a barra de progresso para o score de similaridade.
                        st.dataframe(df_display, 
                                     column_config={"Similaridade": st.column_config.ProgressColumn(
                                         "Similaridade (%)", 
                                         format="%.2f %%", 
                                         min_value=0, 
                                         max_value=100
                                     )})
