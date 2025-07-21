from pydantic import BaseModel

class MatchRequest(BaseModel):
    descricao_candidato: str
    descricao_vaga: str

class MatchResponse(BaseModel):
    match: int
    similaridade: float
