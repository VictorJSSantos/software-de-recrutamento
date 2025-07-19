import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

# Carregar modelo e vetorizador
MODEL_PATH = "models/modelo_match.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def preprocess_text(text):
    from unidecode import unidecode
    import re
    text = unidecode(text.lower())
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

def gerar_embedding(texto):
    texto_limpo = preprocess_text(texto)
    return vectorizer.transform([texto_limpo])

def predict_match(vaga_texto, candidato_texto):
    vaga_vec = gerar_embedding(vaga_texto)
    candidato_vec = gerar_embedding(candidato_texto)

    # Pode concatenar ou fazer média — aqui optamos pela média:
    features = (vaga_vec + candidato_vec) / 2
    prob = model.predict_proba(features)[0][1]
    match = prob >= 0.5
    return match, float(prob)
