import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mplsoccer import PyPizza, FontManager

st.set_page_config(layout="wide")
st.title("PROScout AI")

uploaded_file = st.file_uploader("📂 Carregue um arquivo CSV ou XLSX", type=["csv", "xlsx"])

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
                df[col] = df[col].str.replace(",", ".").astype(float)
            except:
                pass

    # -------------------------------
    # Filtros
    # -------------------------------
    idade_min, idade_max = int(df["Idade"].min()), int(df["Idade"].max())
    idade_sel = st.slider("Idade do jogador", idade_min, idade_max, (idade_min, idade_max))
    
    minplayed_min, minplayed = int(df["Minutos jogados:"].min()), int(df["Minutos jogados:"].max())
    minutesplayed_sel = st.slider("Minutos do jogador na temporada", minplayed_min, minplayed, (minplayed_min, minplayed))
    
    
    posicoes_fixas = ["Goleiro", "Lateral", "Zagueiro", "Volante", 
                      "Meia-Central", "Meia-Ofensivo", "Extremo", "Centroavante"]
    posicao_sel = st.selectbox("Selecione a posição (apenas para o gráfico)", posicoes_fixas)

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


    estilos_validos = estilos_pos.get(posicao_sel, [])
    estilos_escolhidos = st.multiselect("Selecione os estilos", estilos_validos)

    # -------------------------------
    # Mapeamento de estilos → métricas
    # -------------------------------
    metricas_por_estilo = {
        # ----------------
        # Goleiro
        # ----------------
        "Shot Stopper": ["Defesas, %", "Golos sofridos/90", "Golos expectáveis defendidos por 90´"],
        "Sweeper Keeper": ["Saídas/90", "Duelos aéreos/90", "Duelos aéreos ganhos, %"],
        "Distribuidor": ["Passes certos, %", "Passes longos certos, %", "Passes para trás recebidos pelo guarda-redes/90"],

        # ----------------
        # Zagueiro
        # ----------------
        "Defensor": ["Duelos defensivos/90", "Duelos defensivos ganhos, %", "Cortes/90", "Interseções/90", "Faltas/90"],
        "Líder de Defesa": ["Ações defensivas com êxito/90", "Duelos aéreos ganhos, %"],
        "Construtor": ["Passes/90", "Passes certos, %", "Passes progressivos/90", "Passes progressivos certos, %"],
        "Lançador": ["Passes longos/90", "Passes longos certos, %", "Passes em profundidade/90", "Passes em profundidade certos, %"],
        "Dominador Aéreo": ["Duelos aéreos/90", "Duelos aéreos ganhos, %", "Golos de cabeça/90"],

        # ----------------
        # Lateral
        # ----------------
        "Cruzador": ["Cruzamentos/90", "Cruzamentos certos, %", "Passes para a área de penálti/90"],
        "Driblador": ["Dribles/90", "Dribles com sucesso, %", "Acelerações/90"],
        "Desarme": ["Duelos defensivos/90", "Duelos defensivos ganhos, %", "Interseções/90"],
        "Construtor": ["Passes/90", "Passes certos, %", "Passes progressivos/90", "Passes progressivos certos, %"],

        # ----------------
        # Volante
        # ----------------
        "Recuperador": ["Interseções/90", "Duelos defensivos/90", "Duelos defensivos ganhos, %", "Faltas/90"],
        "Box-to-Box": ["Duelos/90", "Interseções/90", "Corridas progressivas/90", "Acelerações/90"],
        "Construtor": ["Passes/90", "Passes certos, %", "Passes progressivos/90", "Passes progressivos certos, %"],
        "Assistente": ["Assistências/90", "Assistências esperadas/90", "Passes chave/90"],

        # ----------------
        # Meia-Central
        # ----------------
        "Construtor": ["Passes/90", "Passes certos, %", "Passes progressivos/90", "Passes progressivos certos, %"],
        "Assistente": ["Assistências/90", "Assistências esperadas/90", "Passes chave/90", "Passes inteligentes/90", "Passes inteligentes certos, %"],
        "Box-to-Box": ["Duelos/90", "Interseções/90", "Corridas progressivas/90", "Acelerações/90"],
        "Driblador": ["Dribles/90", "Dribles com sucesso, %"],
        "Finalizador": ["Golos/90", "Remates/90", "Remates à baliza, %"],

        # ----------------
        # Meia-Ofensivo
        # ----------------
        "Assistente": ["Assistências/90", "Assistências esperadas/90", "Passes chave/90", "Passes inteligentes/90", "Passes inteligentes certos, %"],
        "Finalizador": ["Golos/90", "Remates/90", "Remates à baliza, %", "Golos esperados/90"],
        "Driblador": ["Dribles/90", "Dribles com sucesso, %", "Acelerações/90"],
        "Distribuidor": ["Passes para a frente/90", "Passes para a frente certos, %", "Passes progressivos/90", "Passes progressivos certos, %"],

        # ----------------
        # Extremo
        # ----------------
        "Driblador": ["Dribles/90", "Dribles com sucesso, %", "Acelerações/90"],
        "Cruzador": ["Cruzamentos/90", "Cruzamentos certos, %", "Passes para a área de penálti/90"],
        "Finalizador": ["Golos/90", "Remates/90", "Remates à baliza, %", "Golos esperados/90"],
        "Assistente": ["Assistências/90", "Assistências esperadas/90", "Passes chave/90"],
        "Acelerador": ["Corridas progressivas/90", "Acelerações/90"],

        # ----------------
        # Centroavante
        # ----------------
        "Finalizador": ["Golos/90", "Remates/90", "Remates à baliza, %", "Toques na área/90"],
        "Pressionador": ["Duelos defensivos/90", "Duelos defensivos ganhos, %", "Acções atacantes com sucesso/90"],
        "Dominador Aéreo": ["Duelos aéreos/90", "Duelos aéreos ganhos, %", "Golos de cabeça/90"],
        "Movimentador": ["Acelerações/90", "Corridas progressivas/90", "Passes recebidos/90"],
        "Assistente": ["Assistências/90", "Assistências esperadas/90", "Passes chave/90"]
    }

    # -------------------------------
    # NOVO: Dicionário de Pesos para o Score (Ponderação)
    # -------------------------------
    # Define pesos de 1.0 (menos importante) a 3.0 (mais importante).
    pesos_por_estilo = {
        # ESTILOS GERAIS/REPETIDOS (Usados como fallback se não houver um estilo específico por posição)
        "Construtor": {
            "Passes certos, %": 3.0,
            "Passes progressivos certos, %": 2.5,
            "Passes progressivos/90": 1.5,
            "Passes/90": 1.0,
        },
        "Assistente": {
            "Assistências/90": 3.0,
            "Passes chave/90": 2.5,
            "Assistências esperadas/90": 2.0,
            "Passes inteligentes certos, %": 1.5,
        },
        "Driblador": {
            "Dribles com sucesso, %": 2.5,
            "Dribles/90": 1.5,
            "Acelerações/90": 1.0,
        },
        "Finalizador": {
            "Golos/90": 3.0,
            "Golos esperados/90": 2.5,
            "Remates à baliza, %": 1.5,
            "Remates/90": 1.0,
        },

        # Zagueiro
        "Defensor": {
            "Duelos defensivos ganhos, %": 3.0,
            "Interseções/90": 2.5,
            "Cortes/90": 2.0,
            "Duelos defensivos/90": 1.0,
            "Faltas/90": 1.0,
        },
        "Líder de Defesa": {
            "Duelos aéreos ganhos, %": 3.0,
            "Ações defensivas com êxito/90": 2.0,
        },
        "Lançador": {
            "Passes longos certos, %": 3.0,
            "Passes em profundidade certos, %": 2.5,
            "Passes longos/90": 1.5,
            "Passes em profundidade/90": 1.0,
        },
        
        # Lateral
        "Cruzador": {
            "Cruzamentos certos, %": 3.0,
            "Passes para a área de penálti/90": 2.0,
            "Cruzamentos/90": 1.0,
        },
        "Desarme": {
            "Duelos defensivos ganhos, %": 3.0,
            "Interseções/90": 2.0,
            "Duelos defensivos/90": 1.5,
        },
        
        # Volante
        "Recuperador": {
            "Duelos defensivos ganhos, %": 3.0,
            "Interseções/90": 2.5,
            "Duelos defensivos/90": 1.5,
            "Faltas/90": 1.0,
        },
        "Box-to-Box": {
            "Corridas progressivas/90": 2.5,
            "Interseções/90": 2.0,
            "Duelos/90": 1.5,
            "Acelerações/90": 1.0,
        },
        "Distribuidor": {
            "Passes certos, %": 3.0,
            "Passes curtos / médios precisos, %": 2.5,
            "Passes curtos / médios /90": 1.5,
        },

        # Meia-Ofensivo
        "Distribuidor": { # Sobrescrevendo o Distribuidor geral com foco ofensivo
            "Passes para a frente certos, %": 3.0,
            "Passes progressivos certos, %": 2.5,
            "Passes para a frente/90": 1.5,
            "Passes progressivos/90": 1.0,
        },

        # Extremo
        "Acelerador": {
            "Corridas progressivas/90": 2.5,
            "Acelerações/90": 1.5,
        },

        # Centroavante
        "Finalizador": {
            "Golos/90": 3.0,
            "Toques na área/90": 2.0,
            "Remates à baliza, %": 1.5,
            "Remates/90": 1.0,
        },
        "Pressionador": {
            "Duelos defensivos ganhos, %": 3.0,
            "Acções atacantes com sucesso/90": 2.0,
            "Duelos defensivos/90": 1.0,
        },
        "Dominador Aéreo": {
            "Golos de cabeça/90": 3.0,
            "Duelos aéreos ganhos, %": 2.0,
            "Duelos aéreos/90": 1.0,
        },
        "Movimentador": {
            "Passes recebidos/90": 2.5,
            "Corridas progressivas/90": 1.5,
            "Acelerações/90": 1.0,
        },
        
        # Goleiro
        "Shot Stopper": {
            "Defesas, %": 3.0,
            "Golos expectáveis defendidos por 90´": 2.5,
            "Golos sofridos/90": 1.0,
        },
        "Sweeper Keeper": {
            "Saídas/90": 2.0,
            "Duelos aéreos ganhos, %": 3.0,
            "Duelos aéreos/90": 1.0,
        },
        "Distribuidor": {
            "Passes certos, %": 3.0,
            "Passes longos certos, %": 2.0,
            "Passes para trás recebidos pelo guarda-redes/90": 1.0,
        },
    }

    # -------------------------------
    # KPIs fixos para o radar por posição
    # -------------------------------
    kpis_por_posicao = {
        "Goleiro": {
            "Defendendo": ["Defesas, %", "Golos sofridos/90", "Golos sofridos esperados/90",
                           "Golos expectáveis defendidos por 90´", "Remates sofridos/90", "Jogos sem sofrer golos"],
            "Posse": ["Passes certos, %", "Passes longos/90", "Passes longos certos, %",
                      "Passes para trás recebidos pelo guarda-redes/90", "Saídas/90"],
            "Atacando": []
        },
        "Zagueiro": {
            "Defendendo": ["Ações defensivas com êxito/90", "Duelos defensivos/90", "Duelos defensivos ganhos, %",
                           "Cortes/90", "Cortes de carrinho ajust. à posse", "Remates intercetados/90",
                           "Interseções/90", "Interceções ajust. à posse",
                           "Duelos aéreos/90", "Duelos aéreos ganhos, %"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes para a frente/90", "Passes para a frente certos, %",
                      "Passes laterais/90", "Passes laterais certos, %",
                      "Passes progressivos/90", "Passes progressivos certos, %"],
            "Atacando": ["Golos", "Golos de cabeça/90", "Assistências/90"]
        },
        "Lateral": {
            "Defendendo": ["Duelos defensivos/90", "Duelos defensivos ganhos, %", "Interseções/90", "Cortes/90"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes progressivos/90",
                      "Passes progressivos certos, %", "Corridas progressivas/90"],
            "Atacando": ["Assistências/90", "Assistências esperadas/90",
                         "Cruzamentos/90", "Cruzamentos certos, %",
                         "Cruzamentos do flanco esquerdo/90", "Cruzamentos precisos do flanco esquerdo, %",
                         "Cruzamentos do flanco direito/90", "Cruzamentos precisos do flanco direito, %",
                         "Acelerações/90"]
        },
        "Volante": {
            "Defendendo": ["Duelos defensivos/90", "Duelos defensivos ganhos, %", "Interseções/90", "Faltas/90"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes curtos / médios /90", "Passes curtos / médios precisos, %",
                      "Passes para a frente/90", "Passes para a frente certos, %", "Passes progressivos/90", "Passes progressivos certos, %"],
            "Atacando": ["Assistências/90", "Assistências esperadas/90", "Passes chave/90", "Passes inteligentes/90"]
        },
        "Meia-Ofensivo": {
            "Defendendo": ["Duelos/90", "Duelos ganhos, %"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes chave/90",
                      "Passes para terço final/90", "Passes certos para terço final, %",
                      "Passes para a área de penálti/90", "Passes precisos para a área de penálti, %",
                      "Passes inteligentes/90"],
            "Atacando": ["Golos/90", "Golos esperados/90", "Assistências/90", "Assistências esperadas/90",
                         "Dribles/90", "Dribles com sucesso, %", "Toques na área/90"]
        },
        "Extremo": {
            "Defendendo": ["Duelos ofensivos/90", "Duelos ofensivos ganhos, %"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes progressivos/90", "Passes progressivos certos, %",
                      "Corridas progressivas/90", "Acelerações/90"],
            "Atacando": ["Golos/90", "Golos esperados/90", "Assistências/90", "Assistências esperadas/90",
                         "Cruzamentos/90", "Cruzamentos certos, %", "Dribles/90", "Dribles com sucesso, %",
                         "Toques na área/90"]
        },
        "Centroavante": {
            "Defendendo": ["Ações defensivas com êxito/90", "Duelos aéreos/90", "Duelos aéreos ganhos, %"],
            "Posse": ["Passes/90", "Passes certos, %", "Passes recebidos/90", "Passes longos recebidos/90"],
            "Atacando": ["Golos/90", "Golos sem ser por penálti/90", "Golos esperados/90", "Golos de cabeça/90",
                         "Remates/90", "Remates à baliza, %", "Toques na área/90", "Acelerações/90"]
        }
    }


    # -------------------------------
    # Botão gerar análise
    # -------------------------------
    if st.button("Gerar análise"):

        df_filtrado = df[(df["Idade"] >= idade_sel[0]) & (df["Idade"] <= idade_sel[1])]
        df_filtrado = df[(df["Minutos jogados:"] >= minutesplayed_sel[0]) & (df["Minutos jogados:"] <= minutesplayed_sel[1])]
        if df_filtrado.empty:
            st.warning("Nenhum jogador encontrado com esses filtros.")
        elif not estilos_escolhidos:
            st.warning("Selecione pelo menos um estilo para análise.")
        else:
            # Métricas selecionadas pelos estilos
            metricas = []
            for estilo in estilos_escolhidos:
                metricas.extend(metricas_por_estilo.get(estilo, []))
            metricas_existentes = [m for m in metricas if m in df_filtrado.columns]

            if not metricas_existentes:
                st.warning("Nenhuma métrica válida encontrada no dataset.")
            else:
                df_pos = df_filtrado.copy()
                
                # Métrica para ranqueamento invertido (quanto MENOR o valor, MELHOR a classificação)
                # Ex: 'Golos sofridos/90', 'Faltas/90'
                metricas_negativas = ["Golos sofridos/90", "Faltas/90"] 
                
                # Gerar percentuais para métricas dos estilos
                for col in metricas_existentes:
                    if col in metricas_negativas:
                        # Ranqueamento invertido: asc=False (maior valor é pior, então ranqueia do menor para o maior percentil)
                        df_pos[col + "_pct"] = df_pos[col].rank(pct=True, ascending=False) * 100
                    else:
                        # Ranqueamento normal: asc=True (maior valor é melhor, ranqueia do menor para o maior percentil)
                        df_pos[col + "_pct"] = df_pos[col].rank(pct=True) * 100


                # ---------------------------------------------
                # NOVO CÁLCULO DE SCORE COM PESOS (Ponderação)
                # ---------------------------------------------
                
                # 1. Coletar os pesos para as métricas escolhidas
                pesos_finais = {}
                for estilo in estilos_escolhidos:
                    # Tenta pegar o peso específico do estilo. Se não existir, pega o peso do estilo geral.
                    pesos_estilo = pesos_por_estilo.get(estilo, {})
                    
                    # Se não encontrou pesos específicos para o estilo, tenta o fallback (ex: Estilo genérico)
                    if not pesos_estilo:
                         # Tenta pegar pesos de estilos genéricos/comuns (ex: 'Finalizador' genérico)
                         # Isso pode ser refinado para uma lógica mais complexa, mas aqui simplificamos a busca.
                         for key, value in pesos_por_estilo.items():
                             if key == estilo:
                                 pesos_estilo = value
                                 break

                    # Adiciona os pesos das métricas
                    for metrica, peso in pesos_estilo.items():
                        if metrica in metricas_existentes:
                            pesos_finais[metrica] = peso 
                        
                # 2. Aplicar a Média Ponderada
                df_pos["Score Ponderado"] = 0.0
                soma_pesos = sum(pesos_finais.values())
                
                if soma_pesos > 0:
                    for metrica, peso in pesos_finais.items():
                        # Score Ponderado += (Percentil * Peso)
                        df_pos["Score Ponderado"] += df_pos[metrica + "_pct"] * peso
                        
                    # Score Final = Soma Ponderada / Soma dos Pesos
                    df_pos["Score"] = df_pos["Score Ponderado"] / soma_pesos
                else:
                    # Fallback para média simples (original) se não houver pesos definidos
                    df_pos["Score"] = df_pos[[c+"_pct" for c in metricas_existentes]].mean(axis=1)

                df_final = df_pos.sort_values(by="Score", ascending=False)
                # ---------------------------------------------
                # FIM DO NOVO CÁLCULO DE SCORE
                # ---------------------------------------------


                st.dataframe(df_final[["Jogador", "Equipa", "Idade", "Score"] + metricas_existentes].round(1))


                # -------------------------------
                # Radar do melhor jogador (percentual para todos KPIs do radar)
                # -------------------------------
                # Gerar percentuais para todas métricas do radar
                todas_metricas_radar = []
                for kpis_pos in kpis_por_posicao.values():
                    for grupo_metrica in kpis_pos.values():
                        todas_metricas_radar.extend(grupo_metrica)
                todas_metricas_radar = list(set([m for m in todas_metricas_radar if m in df.columns]))
                
                # Aplicar ranqueamento para o radar (necessário recalcular ranques invertidos)
                for col in todas_metricas_radar:
                    if col in metricas_negativas:
                         df_final[col + "_pct"] = df_final[col].rank(pct=True, ascending=False) * 100
                    else:
                        df_final[col + "_pct"] = df_final[col].rank(pct=True) * 100

                top_player = df_final.iloc[0]
                st.subheader(f"Jogador Sugerido - {top_player['Jogador']} ({posicao_sel})")

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
                        f"{top_player['Jogador']} - {top_player['Equipa']} ({', '.join(estilos_escolhidos)})",
                        size=12, ha="center", fontproperties=font_bold.prop, color="#000000"
                    )

                    st.pyplot(fig)
                else:
                    st.warning("Não há métricas disponíveis para o radar desta posição.")