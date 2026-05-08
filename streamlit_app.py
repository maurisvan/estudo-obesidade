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

# --- ESTILO CSS CUSTOMIZADO (Fundo Branco e Fontes Pretas) ---
st.markdown("""
    <style>
        .stApp { background-color: #FFFFFF; }
        h1, h2, h3, p, span, label, .stMarkdown { color: #000000 !important; font-family: 'Segoe UI', sans-serif; }
        
        /* Campos de Entrada com destaque */
        .stTextInput div div input, .stSelectbox div div select, .stNumberInput div div input {
            background-color: #f0f2f6 !important; 
            border: 1px solid #d1d5db !important;
            color: #000000 !important;
            border-radius: 8px !important;
        }

        /* Botão Principal */
        button[kind="primary"] {
            background-color: #007bff !important;
            color: white !important;
            font-weight: bold !important;
            width: 100%;
            border-radius: 8px !important;
        }

        /* Métrica e Expanders */
        div[data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 12px !important;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05) !important;
        }
        div[data-testid="stExpander"] {
            background-color: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 12px !important;
        }
        [data-testid="stSidebar"] { background-color: #f8f9fa !important; }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DE MODELOS ---
# Os nomes abaixo devem ser EXATAMENTE iguais aos arquivos no seu GitHub
MODEL_PATH = 'modelo_obesidade.pkl'
LE_PATH = 'label_encoder.pkl'

@st.cache_resource
def carregar_recursos():
    # Carregamento direto para identificar erros de versão ou caminho
    modelo = joblib.load(MODEL_PATH)
    encoder = joblib.load(LE_PATH)
    return modelo, encoder

# Se os arquivos estiverem no GitHub, o app deve carregar aqui
try:
    pipeline, le = carregar_recursos()
    modelo_carregado = True
except Exception as e:
    st.error(f"Erro ao carregar arquivos do modelo: {e}")
    modelo_carregado = False

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/bb/Logo_do_SUS.svg", width=100)
    st.title("Configurações")
    if modelo_carregado:
        st.success("✅ Modelo carregado!")
    else:
        st.error("❌ Modelo não encontrado na raiz do projeto.")
    
    st.info("Sistema de Triagem v1.0")

# --- CONTEÚDO PRINCIPAL ---
st.title("🏥 Apoio ao Diagnóstico de Obesidade")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios"])

with tab1:
    with st.form("form_paciente"):
        col_form1, col_form2 = st.columns([2, 1])
        
        with col_form1:
            st.subheader("📋 Coleta de Dados")
            
            with st.expander("👤 Dados Biométricos", expanded=True):
                c1, c2, c3 = st.columns(3)
                genero_v = c1.selectbox("Gênero", ['Masculino', 'Feminino'])
                idade = c2.number_input("Idade", 1, 120, 24)
                hist_fam = c3.selectbox("Histórico Familiar?", ['Sim', 'Não'])
                altura = c1.slider("Altura (m)", 0.50, 2.30, 1.70)
                peso = c2.slider("Peso (kg)", 10.0, 250.0, 80.0)

            with st.expander("🥗 Hábitos Alimentares"):
                c4, c5 = st.columns(2)
                favc = c4.radio("Comida calórica frequente?", ['Sim', 'Não'], horizontal=True)
                caec = c5.selectbox("Come entre refeições?", ['Às vezes', 'Frequentemente', 'Sempre', 'Não'])
                fcvc = st.select_slider("Consumo de vegetais", options=[1, 2, 3])
                ncp = st.select_slider("Refeições principais/dia", options=[1, 2, 3, 4])

            with st.expander("🏃 Estilo de Vida"):
                c6, c7 = st.columns(2)
                faf = c6.select_slider("Atividade física (Freq)", options=[0, 1, 2, 3])
                ch2o = c7.select_slider("Água diária (Litros)", options=[1, 2, 3])
                mtrans = st.selectbox("Meio de transporte", ['Transporte Público', 'Caminhada', 'Carro', 'Moto', 'Bicicleta'])

        with col_form2:
            st.subheader("🚀 Processamento")
            btn_diagnostico = st.form_submit_button("GERAR DIAGNÓSTICO", type="primary")
            
            if btn_diagnostico:
                if modelo_carregado:
                    # Mapeamentos para as colunas que o seu modelo espera
                    mapa_genero = {'Masculino': 'Male', 'Feminino': 'Female'} 
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

                    # Predição
                    pred_codificada = pipeline.predict(df_input)
                    resultado_raw = le.inverse_transform(pred_codificada)[0]
                    
                    st.metric("IMC Calculado", f"{imc_input:.2f}")
                    
                    # Cor baseada no resultado
                    color = "#ef4444" if "Obesity" in resultado_raw else "#f59e0b" if "Overweight" in resultado_raw else "#10b981"
                    st.markdown(f"""
                        <div style="background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center;">
                            <h2 style="margin:0; color:white !important;">{resultado_raw.replace('_', ' ')}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("O modelo não pôde ser utilizado porque os arquivos .pkl não foram carregados.")

with tab2:
    st.header("📊 Painel Geral")
    m1, m2, m3 = st.columns(3)
    m1.metric("Pacientes na Base", "2.111")
    m2.metric("Peso Médio", "86,6 kg")
    m3.metric("Idade Média", "24,3")
    
    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        fig_p = px.pie(names=['Obesidade', 'Sobrepeso', 'Normal'], values=[45, 30, 25], hole=0.4, title="Distribuição de Casos")
        fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_p, use_container_width=True)
    with g2:
        d_transp = {'Meio': ['Público', 'Auto', 'Caminhada'], 'Qtd': [1558, 463, 88]}
        fig_t = px.bar(d_transp, x='Meio', y='Qtd', title="Mobilidade", color_discrete_sequence=['#007bff'])
        fig_t.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_t, use_container_width=True)

with tab3:
    st.link_button("📂 Acessar Looker Studio", "https://lookerstudio.google.com/", use_container_width=True)
