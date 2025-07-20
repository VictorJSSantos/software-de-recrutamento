from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Summary


from app.schema import MatchRequest, MatchResponse
from app.model import predict_match


app = FastAPI()
Instrumentator().instrument(app).expose(app)
REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")


@app.post("/predict", response_model=MatchResponse)
@REQUEST_TIME.time()
def predict(request: MatchRequest):
    vaga_texto = f"{request.vaga.titulo} {request.vaga.descricao}"
    candidato_texto = f"{request.candidato.nome} {request.candidato.resumo}"

    match, prob = predict_match(vaga_texto, candidato_texto)
    return {"match": match, "probabilidade": prob}
