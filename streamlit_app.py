import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import streamlit.components.v1 as components

# 1. Configuração da Página (Título da Aba e Favicon)
st.set_page_config(
    page_title="Sistema de Diagnóstico - Obesidade", 
    page_icon="logo_sus.png", 
    layout="wide"
)

# Caminhos dos arquivos
MODEL_PATH = 'modelo_obesidade.pkl'
LE_PATH = 'label_encoder.pkl'
DATA_PATH = 'Obesity.csv'

# 2. Função para carregar o modelo e o encoder
@st.cache_resource
def carregar_recursos():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        st.error("Erro: Arquivos 'modelo_obesidade.pkl' ou 'label_encoder.pkl' não encontrados.")
        return None, None
    try:
        modelo = joblib.load(MODEL_PATH)
        encoder = joblib.load(LE_PATH)
        return modelo, encoder
    except Exception as e:
        st.error(f"Erro técnico ao carregar recursos: {e}")
        return None, None

pipeline, le = carregar_recursos()

# 3. Cabeçalho Principal com Logo do SUS
col_logo, col_titulo = st.columns([1, 5])

with col_logo:
    # Busca o arquivo logo_sus.png na raiz do seu diretório
    st.image("logo_sus.png", width=120)

with col_titulo:
    st.title("Sistema de Apoio ao Diagnóstico de Obesidade")
    st.subheader("Rede SUS - Sistema Único de Saúde")

st.markdown("---")

# Definição das Abas
tab1, tab2, tab3 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios e Insights"])

# --- TAB 1: FORMULÁRIO E PREDIÇÃO ---
with tab1:
    st.header("Formulário do Paciente")
    col1, col2, col3 = st.columns(3)

    # Dicionários de Tradução (Visual PT -> Modelo EN)
    mapa_genero = {'Masculino': 'Female', 'Feminino': 'Male'} 
    mapa_sim_nao = {'Sim': 'yes', 'Não': 'no'}
    mapa_frequencia = {'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always', 'Não': 'no'}
    mapa_transporte = {
        'Transporte Público': 'Public_Transportation', 'Caminhada': 'Walking', 
        'Carro': 'Automobile', 'Moto': 'Motorbike', 'Bicicleta': 'Bike'
    }

    with col1:
        genero_v = st.selectbox("Gênero", list(mapa_genero.keys()))
        idade = st.number_input("Idade", 1, 120, 24)
        altura = st.number_input("Altura (m)", 0.5, 2.5, 1.70)
        peso = st.number_input("Peso (kg)", 10.0, 300.0, 86.59)
        hist_fam = st.selectbox("Histórico Familiar de Sobrepeso?", list(mapa_sim_nao.keys()))

    with col2:
        favc = st.selectbox("Consome comida calórica frequentemente?", list(mapa_sim_nao.keys()))
        fcvc = st.slider("Frequência de consumo de vegetais (1-3)", 1, 3, 2)
        ncp = st.slider("Número de refeições principais", 1, 4, 3)
        caec = st.selectbox("Come entre refeições?", list(mapa_frequencia.keys()))
        smoke = st.selectbox("Fumante?", list(mapa_sim_nao.keys()))

    with col3:
        ch2o = st.slider("Consumo de água diário (1-3L)", 1, 3, 2)
        scc = st.selectbox("Monitora calorias ingeridas?", list(mapa_sim_nao.keys()))
        faf = st.slider("Frequência de atividade física (0-3)", 0, 3, 1)
        tue = st.slider("Tempo usando dispositivos (0-2)", 0, 2, 1)
        calc = st.selectbox("Consumo de álcool", list(mapa_frequencia.keys()))
        mtrans = st.selectbox("Meio de transporte principal", list(mapa_transporte.keys()))

    if st.button("Realizar Diagnóstico"):
        if pipeline and le:
            # Cálculo do IMC
            imc_input = peso / (altura ** 2)
            
            # DataFrame de entrada
            df_input = pd.DataFrame({
                'genero': [mapa_genero[genero_v]],
                'idade': [idade],
                'altura_m': [altura],
                'peso_kg': [peso],
                'historia_familiar_sobrepeso': [mapa_sim_nao[hist_fam]],
                'come_comida_calorica_freq': [mapa_sim_nao[favc]],
                'freq_consumo_vegetais': [fcvc],
                'num_refeicoes_principais': [ncp],
                'come_entre_refeicoes': [mapa_frequencia[caec]],
                'fumante': [mapa_sim_nao[smoke]],
                'consumo_agua_litros': [ch2o],
                'monitora_calorias': [mapa_sim_nao[scc]],
                'freq_atividade_fisica': [faf],
                'tempo_uso_dispositivos': [tue],
                'freq_consumo_alcool': [mapa_frequencia[calc]],
                'meio_transporte': [mapa_transporte[mtrans]],
                'imc': [imc_input]
            })

            try:
                # Predição
                pred_codificada = pipeline.predict(df_input)
                resultado_raw = le.inverse_transform(pred_codificada)[0]

                # Normalização do resultado
                def normalize(level):
                    if level == 'Insufficient_Weight': return "Abaixo do peso"
                    elif level == 'Normal_Weight': return "Peso normal"
                    elif level in ['Overweight_Level_I', 'Overweight_Level_II']: return "Sobrepeso"
                    else: return "Obeso"

                resultado_final = normalize(resultado_raw)

                # Exibição
                st.success(f"### Resultado: {resultado_final}")
                st.info(f"**Classificação Detalhada:** {resultado_raw.replace('_', ' ')}")
                st.info(f"**IMC Calculado:** {imc_input:.2f}")

            except Exception as e:
                st.error(f"Erro na predição: {e}")

# --- TAB 2: DASHBOARD NATIVO ---
with tab2:
    st.header("📊 Painel de Indicadores Locais")
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pacientes", "2.111")
    k2.metric("Peso Médio", "86,59 kg", delta="Estável")
    k3.metric("Idade Média", "24 anos")
    k4.metric("Fator de Risco", "81%", help="Pacientes com histórico familiar")

    st.markdown("---")
    
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Distribuição por Categoria")
        fig_p = px.pie(
            names=['Obesidade I', 'Obesidade III', 'Obesidade II', 'Sobrepeso II', 'Sobrepeso I', 'Peso Normal', 'Abaixo do Peso'],
            values=[16.6, 15.3, 14.1, 13.7, 13.7, 13.6, 12.9],
            hole=0.4, color_discrete_sequence=px.colors.qualitative.T10
        )
        st.plotly_chart(fig_p, use_container_width=True)
        
    with g2:
        st.subheader("Transporte e Sedentarismo")
        d_transp = {'Meio': ['Público', 'Automóvel', 'Caminhada'], 'Qtd': [1558, 463, 88]}
        fig_t = px.bar(d_transp, x='Meio', y='Qtd', color='Meio', text_auto=True)
        st.plotly_chart(fig_t, use_container_width=True)

# --- TAB 3: RELATÓRIO LOOKER STUDIO ---
with tab3:
    st.header("📝 Relatórios e Insights")
    
    # O link do botão pode continuar o original para abrir em tela cheia
    st.link_button("🚀 Abrir Relatório Completo no Looker Studio", 
                   "https://lookerstudio.google.com/reporting/a2861988-83f6-4037-ab24-5b046d3b76fa")

    st.markdown("---")
    
    st.subheader("Visualização Rápida")
    
    # AQUI ESTÁ A MUDANÇA: Adicionamos o /embed/ e removemos o /reporting/ na URL do iframe
    st.components.v1.iframe(
        "https://lookerstudio.google.com/embed/reporting/a2861988-83f6-4037-ab24-5b046d3b76fa/page/1M",
        height=700,
        scrolling=True
    )

    st.info("💡 Insight: O histórico familiar e o sedentarismo no transporte são os principais fatores identificados.")
