from fastapi import FastAPI
from app.schema import MatchRequest, MatchResponse
from app.model import predict_match

app = FastAPI()

@app.post("/predict", response_model=MatchResponse)
def predict(request: MatchRequest):
    vaga_texto = f"{request.vaga.titulo} {request.vaga.descricao}"
    candidato_texto = f"{request.candidato.nome} {request.candidato.resumo}"
    
    match, prob = predict_match(vaga_texto, candidato_texto)
    return {"match": match, "probabilidade": prob}
