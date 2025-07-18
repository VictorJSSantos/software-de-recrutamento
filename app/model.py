import os
import joblib
import pandas as pd

MODEL_PATH = "models/modelo_match.pkl"

# Carrega o modelo salvo do disco
def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Treine o modelo primeiro.")
    return joblib.load(MODEL_PATH)

# Realiza a predição com o modelo carregado
def prever(modelo, dados: pd.DataFrame):
    if not isinstance(dados, pd.DataFrame):
        raise ValueError("Os dados de entrada devem ser um DataFrame do pandas.")
    
    # Garante que não tem colunas extras como id_candidato ou id_vaga
    colunas_para_remover = [col for col in ["id_candidato", "id_vaga", "match"] if col in dados.columns]
    dados = dados.drop(columns=colunas_para_remover, errors="ignore")
    
    # Faz a predição
    predicoes = modelo.predict(dados)
    probabilidades = modelo.predict_proba(dados)[:, 1]  # Probabilidade da classe 1 (match)
    
    return predicoes.tolist(), probabilidades.tolist()
