from pydantic import BaseModel
from typing import Literal


class ApplicantJobMatchInput(BaseModel):
    nivel_academico: str
    nivel_ingles: str
    nivel_espanhol: str
    area_atuacao: str
    cv: str  # texto do currículo (PT)
    descricao_vaga: str  # principais atividades + competências técnicas/comportamentais


class ApplicantJobMatchOutput(BaseModel):
    prediction: Literal[0, 1]  # 0 = não contratado, 1 = contratado
    score: float  # probabilidade (ex: 0.82)