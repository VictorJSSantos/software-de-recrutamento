from fastapi import FastAPI
from app.schema import MatchRequest, MatchResponse
from pipeline.preprocessing import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

@app.post("/predict", response_model=MatchResponse)
def predict(data: MatchRequest):
    # Pré-processar os textos
    candidato = preprocess_text(data.descricao_candidato)
    vaga = preprocess_text(data.descricao_vaga)

    # Vetorização com TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([candidato, vaga])

    # Similaridade cosseno
    sim = cosine_similarity(vectors[0], vectors[1])[0][0]
    match = int(sim > 0.5)  # limiar ajustável

    return MatchResponse(match=match, similaridade=round(sim, 2))
