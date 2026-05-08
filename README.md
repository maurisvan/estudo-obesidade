
🏥 Sistema de Diagnóstico de Obesidade - Hospital Vita Nova
Este projeto é uma aplicação web interativa desenvolvida com Streamlit para auxiliar no diagnóstico clínico de níveis de obesidade. O sistema combina predições de Machine Learning em tempo real com análises visuais de dados.

🚀 Funcionalidades
O sistema está dividido em três módulos principais:

🔮 Predição Clínica: Interface para entrada de dados do paciente (hábitos alimentares, físico e histórico). O modelo processa os dados e retorna o diagnóstico e o IMC calculado.

📊 Dashboard Analítico: Visualização de indicadores-chave da clínica (KPIs) e gráficos de distribuição de perfis de pacientes utilizando Plotly.

📝 Relatórios e Insights: Integração com Looker Studio via iframe para consultas detalhadas e relatórios dinâmicos externos.

https://lookerstudio.google.com/u/0/reporting/29f80ed0-090c-437e-a0e8-a3fd3b00e5be/page/2V5oF

🛠️ Tecnologias Utilizadas
Linguagem: Python 3.x

Interface Web: Streamlit

https://p-s-tech-data-analytics-fase4-76epkgdr4mny4khaudkkse.streamlit.app/

Machine Learning: Scikit-Learn (Pipeline e Label Encoding)

Processamento de Dados: Pandas & Joblib

Visualização: Plotly Express & Looker Studio

📂 Estrutura de Arquivos
app.py: Código principal da aplicação.

modelo_obesidade.pkl: Pipeline do modelo de classificação treinado.

label_encoder.pkl: Codificador para tratamento das variáveis categóricas.

Obesity.csv: Base de dados utilizada para os insights.
