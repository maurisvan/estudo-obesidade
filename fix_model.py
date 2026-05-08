import joblib
import sklearn
print(f"Sua versão atual do sklearn é: {sklearn.__version__}")

# Tente carregar o modelo e salvar novamente para atualizar a 'receita' do pickle
try:
    # Carrega o modelo (se der erro aqui, vamos precisar de outra abordagem)
    modelo = joblib.load('modelo_obesidade.pkl')
    encoder = joblib.load('label_encoder.pkl')
    
    # Salva novamente por cima ou com outro nome
    joblib.dump(modelo, 'modelo_obesidade_v2.pkl')
    joblib.dump(encoder, 'label_encoder_v2.pkl')
    
    print("Sucesso! Modelos re-salvados com a versão atual.")
except Exception as e:
    print(f"Não foi possível converter: {e}")