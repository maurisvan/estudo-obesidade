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

# --- ESTILO CSS CUSTOMIZADO (Design Clean & Profissional) ---
st.markdown("""
    <style>
        /* Fundo principal totalmente branco */
        .stApp {
            background-color: #FFFFFF;
        }

        /* Títulos e textos em preto puro para legibilidade */
        h1, h2, h3, p, span, label, .stMarkdown {
            color: #000000 !important;
            font-family: 'Segoe UI', Roboto, sans-serif;
        }

        /* Customização dos campos de entrada (Inputs e Selects) */
        .stTextInput div div input, .stSelectbox div div select, .stNumberInput div div input, .stMultiSelect div div {
            background-color: #f0f2f6 !important; 
            border: 1px solid #d1d5db !important;
            color: #000000 !important;
            border-radius: 8px !important;
        }

        /* Cards de Métricas (IMC e Indicadores) */
        div[data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            padding: 15px !important;
            border-radius: 12px !important;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05) !important;
        }

        /* Estilização dos Expanders (Sanfonas) */
        div[data-testid="stExpander"] {
            background-color: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 12px !important;
            margin-bottom: 10px !important;
        }

        /* Botão Principal Estilo SUS */
        button[kind="primary"] {
            background-color: #007bff !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 2rem !important;
            font-weight: bold !important;
            width: 100%;
        }

        /* Sidebar com leve contraste */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e5e7eb;
        }

        /* Ajuste de abas */
        button[data-baseweb="tab"] p {
            font-size: 18px !important;
            font-weight: 600 !important;
            color: #4b5563 !important;
        }
        button[aria-selected="true"] p {
            color: #007bff !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DE MODELOS ---
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
        st.success("Modelo pronto para uso")
    else:
        st.error("Erro: Arquivos .pkl não encontrados")
    
    st.info("Sistema de Triagem Metabólica v1.0")

# --- CONTEÚDO PRINCIPAL ---
st.title("🏥 Apoio ao Diagnóstico de Obesidade")
st.markdown("Utilize os campos abaixo para realizar a predição clínica baseada em hábitos de vida.")

tab1, tab2, tab3 = st.tabs(["🔮 Predição Clínica", "📊 Dashboard Analítico", "📝 Relatórios"])

with tab1:
    with st.form("form_paciente"):
        col_form1, col_form2 = st.columns([2, 1])
        
        with col_form1:
            st.subheader("📋 Coleta de Dados")
            
            exp1 = st.expander("👤 Dados Biométricos", expanded=True)
            with exp1:
                c1, c2, c3 = st.columns(3)
                genero_v = c1.selectbox("Gênero", ['Masculino', 'Feminino'])
                idade = c2.number_input("Idade", 1, 120, 24)
                hist_fam = c3.selectbox("Histórico Familiar?", ['Sim', 'Não'])
                
                altura = c1.slider("Altura (m)", 0.50, 2.30, 1.70)
                peso = c2.slider("Peso (kg)", 10.0, 250.0, 80.0)

            exp2 = st.expander("🥗 Hábitos Alimentares")
            with exp2:
                c4, c5 = st.columns(2)
                favc = c4.radio("Comida calórica frequente?", ['Sim', 'Não'], horizontal=True)
                caec = c5.selectbox("Come entre refeições?", ['Às vezes', 'Frequentemente', 'Sempre', 'Não'])
                fcvc = st.select_slider("Consumo de vegetais", options=[1, 2, 3])
                ncp = st.select_slider("Refeições principais/dia", options=[1, 2, 3, 4])

            exp3 = st.expander("🏃 Estilo de Vida")
            with exp3:
                c6, c7 = st.columns(2)
                faf = c6.select_slider("Atividade física (Freq)", options=[0, 1, 2, 3])
                ch2o = c7.select_slider("Água diária (Litros)", options=[1, 2, 3])
                mtrans = st.selectbox("Meio de transporte", ['Transporte Público', 'Caminhada', 'Carro', 'Moto', 'Bicicleta'])

        with col_form2:
            st.subheader("🚀 Processamento")
            st.write("Clique no botão abaixo para processar os dados através do modelo de IA.")
            btn_diagnostico = st.form_submit_button("GERAR DIAGNÓSTICO", type="primary")
            
            if btn_diagnostico:
                if pipeline:
                    # Mapeamentos para o modelo
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

                    pred_codificada = pipeline.predict(df_input)
                    resultado_raw = le.inverse_transform(pred_codificada)[0]
                    
                    st.markdown("---")
                    st.metric("IMC Calculado", f"{imc_input:.2f}")
                    
                    color = "#ef4444" if "Obesity" in resultado_raw else "#f59e0b" if "Overweight" in resultado_raw else "#10b981"
                    st.markdown(f"""
                        <div style="background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center;">
                            <small>CLASSIFICAÇÃO IDENTIFICADA</small>
                            <h2 style="margin:0; color:white !important;">{resultado_raw.replace('_', ' ')}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Erro Crítico: Modelo não carregado. Verifique os logs.")

with tab2:
    st.header("📊 Painel de Indicadores Gerais")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pacientes na Base", "2.111", "+12%")
    m2.metric("Peso Médio", "86,6 kg")
    m3.metric("Idade Média", "24,3")
    m4.metric("Aderência SUS", "94%")
    
    st.markdown("---")
    g1, g2 = st.columns(2)
    
    # Ajuste de transparência nos gráficos para o novo fundo branco
    with g1:
        fig_p = px.pie(names=['Obesidade', 'Sobrepeso', 'Normal'], values=[45, 30, 25], hole=0.5, title="Distribuição de Casos")
        fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_p, use_container_width=True)
    with g2:
        d_transp = {'Meio': ['Público', 'Auto', 'Caminhada'], 'Qtd': [1558, 463, 88]}
        fig_t = px.bar(d_transp, x='Meio', y='Qtd', title="Meios de Mobilidade", color_discrete_sequence=['#007bff'])
        fig_t.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_t, use_container_width=True)

with tab3:
    st.info("Os relatórios detalhados são gerados via Looker Studio para integração de dados em tempo real.")
    st.link_button("📂 Acessar Relatório Completo", "https://lookerstudio.google.com/", use_container_width=True)
