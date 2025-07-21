from fastapi import FastAPI
from app.schema import MatchRequest
import joblib
import os
import pandas as pd
from pipeline.preprocessing import preprocess_text  # certifique-se de que essa função está disponível
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Carrega modelo e vetorizador
modelo = joblib.load("models/modelo_match.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.post("/predict")
def predict_match(request: MatchRequest):
    try:
        # Concatena os campos do candidato
        descricao_candidato = f"{request.nivel_academico} {request.nivel_ingles} {request.nivel_espanhol} {request.area_atuacao} {request.cv}"

        # Pré-processa
        candidato_limpo = preprocess_text(descricao_candidato)
        vaga_limpa = preprocess_text(request.descricao_vaga)

        # Vetoriza
        candidato_vec = vectorizer.transform([candidato_limpo])
        vaga_vec = vectorizer.transform([vaga_limpa])

        # Calcula similaridade
        similaridade = cosine_similarity(candidato_vec, vaga_vec)[0][0]

        # Cria DataFrame com mesma estrutura usada no treino
        df_pred = pd.DataFrame([{
            "descricao_completa": f"{descricao_candidato} {request.descricao_vaga}",
            "similaridade": similaridade
        }])

        # Faz predição
        match = modelo.predict(df_pred)[0]

        return {
            "match": int(match),
            "similaridade": float(similaridade)
        }

    except Exception as e:
        return {"erro": str(e)}
