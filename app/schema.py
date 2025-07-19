from pydantic import BaseModel

class VagaInput(BaseModel):
    titulo: str
    descricao: str

class CandidatoInput(BaseModel):
    nome: str
    resumo: str

class MatchRequest(BaseModel):
    vaga: VagaInput
    candidato: CandidatoInput

class MatchResponse(BaseModel):
    match: bool
    probabilidade: float
