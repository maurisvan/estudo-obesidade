import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# 1. Configuração da Página
st.set_page_config(
    page_title="Diagnóstico de Obesidade | SUS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILO CSS CUSTOMIZADO ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { background-color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Caminhos dos arquivos
MODEL_PATH = 'modelo_obesidade.pkl'
LE_PATH = 'label_encoder.pkl'

@st.cache_resource
def carregar_recursos():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        return None, None
    try:
        modelo = joblib.load(MODEL_PATH)
        encoder = joblib.load(LE_PATH)
        return modelo, encoder
    except:
        return None, None

pipeline, le = carregar_recursos()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/bb/Logo_do_SUS.svg", width=100)
    st.title("Configurações")
    if pipeline:
        st.success("Modelo carregado com sucesso!")
    else:
        st.error("Modelo não encontrado.")
    
    st.info("Este sistema auxilia profissionais de saúde no rastreamento de riscos metabólicos.")

# --- TÍTULO PRINCIPAL ---
st.title("🏥 Sistema de Apoio ao Diagnóstico de Obesidade")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios"])

with tab1:
    # Organizando em containers para não poluir a tela
    with st.form("form_paciente"):
        col_form1, col_form2 = st.columns([2, 1])
        
        with col_form1:
            st.subheader("📋 Coleta de Dados")
            
            exp1 = st.expander("👤 Dados Biométricos", expanded=True)
            with exp1:
                c1, c2, c3 = st.columns(3)
                genero_v = c1.selectbox("Gênero", ['Masculino', 'Feminino'])
                idade = c2.number_input("Idade", 1, 120, 24)
                hist_fam = c3.selectbox("Histórico Familiar de Sobrepeso?", ['Sim', 'Não'])
                
                altura = c1.slider("Altura (m)", 0.50, 2.30, 1.70)
                peso = c2.slider("Peso (kg)", 10.0, 250.0, 80.0)

            exp2 = st.expander("🥗 Hábitos Alimentares")
            with exp2:
                c4, c5 = st.columns(2)
                favc = c4.radio("Consome comida calórica frequentemente?", ['Sim', 'Não'], horizontal=True)
                caec = c5.selectbox("Come entre refeições?", ['Às vezes', 'Frequentemente', 'Sempre', 'Não'])
                fcvc = st.select_slider("Frequência de consumo de vegetais", options=[1, 2, 3], help="1: Baixo, 3: Alto")
                ncp = st.select_slider("Número de refeições principais", options=[1, 2, 3, 4])

            exp3 = st.expander("🏃 Estilo de Vida e Mobilidade")
            with exp3:
                c6, c7 = st.columns(2)
                faf = c6.select_slider("Atividade física (0-3)", options=[0, 1, 2, 3])
                ch2o = c7.select_slider("Água diária (1-3L)", options=[1, 2, 3])
                mtrans = st.selectbox("Meio de transporte principal", ['Transporte Público', 'Caminhada', 'Carro', 'Moto', 'Bicicleta'])

        with col_form2:
            st.subheader("🚀 Ação")
            st.write("Verifique se os dados estão corretos antes de processar.")
            btn_diagnostico = st.form_submit_button("GERAR DIAGNÓSTICO", use_container_width=True, type="primary")
            
            if btn_diagnostico:
                if pipeline:
                    # Lógica de processamento (Mapas)
                    mapa_genero = {'Masculino': 'Female', 'Feminino': 'Male'} 
                    mapa_sim_nao = {'Sim': 'yes', 'Não': 'no'}
                    mapa_frequencia = {'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always', 'Não': 'no'}
                    mapa_transporte = {'Transporte Público': 'Public_Transportation', 'Caminhada': 'Walking', 'Carro': 'Automobile', 'Moto': 'Motorbike', 'Bicicleta': 'Bike'}

                    imc_input = peso / (altura ** 2)
                    
                    df_input = pd.DataFrame({
                        'genero': [mapa_genero[genero_v]], 'idade': [idade], 'altura_m': [altura],
                        'peso_kg': [peso], 'historia_familiar_sobrepeso': [mapa_sim_nao[hist_fam]],
                        'come_comida_calorica_freq': [mapa_sim_nao[favc]], 'freq_consumo_vegetais': [fcvc],
                        'num_refeicoes_principais': [ncp], 'come_entre_refeicoes': [mapa_frequencia[caec]],
                        'fumante': ['no'], 'consumo_agua_litros': [ch2o], 'monitora_calorias': ['no'],
                        'freq_atividade_fisica': [faf], 'tempo_uso_dispositivos': [1],
                        'freq_consumo_alcool': ['Sometimes'], 'meio_transporte': [mapa_transporte[mtrans]], 'imc': [imc_input]
                    })

                    pred_codificada = pipeline.predict(df_input)
                    resultado_raw = le.inverse_transform(pred_codificada)[0]
                    
                    # Exibição de Resultado Estilizada
                    st.markdown("### 🩺 Resultado")
                    st.metric("IMC Calculado", f"{imc_input:.2f}")
                    
                    color = "#ef4444" if "Obesity" in resultado_raw else "#f59e0b" if "Overweight" in resultado_raw else "#10b981"
                    st.markdown(f"""
                        <div style="background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center;">
                            <h2 style="margin:0;">{resultado_raw.replace('_', ' ')}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Modelo não carregado.")

with tab2:
    st.header("📊 Painel de Indicadores Gerais")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pacientes", "2.111", "+12%")
    m2.metric("Peso Médio", "86,6 kg")
    m3.metric("Idade Média", "24,3")
    m4.metric("Aderência SUS", "94%")
    
    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        fig_p = px.pie(names=['Obesidade', 'Sobrepeso', 'Normal'], values=[45, 30, 25], hole=0.5, title="Distribuição Simplificada")
        st.plotly_chart(fig_p, use_container_width=True)
    with g2:
        d_transp = {'Meio': ['Público', 'Auto', 'Caminhada'], 'Qtd': [1558, 463, 88]}
        fig_t = px.bar(d_transp, x='Meio', y='Qtd', title="Mobilidade dos Pacientes", color_discrete_sequence=['#007bff'])
        st.plotly_chart(fig_t, use_container_width=True)

with tab3:
    st.link_button("📂 Abrir Looker Studio", "https://lookerstudio.google.com/...", use_container_width=True)
    st.components.v1.iframe("https://lookerstudio.google.com/embed/reporting/...", height=800)