import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from app.schema import ApplicantJobMatchInput


def load_model(model_path: str = "model/model.joblib") -> Pipeline:
    """
    Carrega o modelo treinado salvo com joblib.
    """
    model = joblib.load(model_path)
    return model


def predict(model: Pipeline, input_data: ApplicantJobMatchInput) -> tuple[int, float]:
    """
    Recebe o input do candidato + vaga, retorna (classe, probabilidade).
    """
    data = [{
        "nivel_academico": input_data.nivel_academico,
        "nivel_ingles": input_data.nivel_ingles,
        "nivel_espanhol": input_data.nivel_espanhol,
        "area_atuacao": input_data.area_atuacao,
        "cv": input_data.cv,
        "descricao_vaga": input_data.descricao_vaga
    }]

    prediction_proba = model.predict_proba(data)[0]
    prediction = int(np.argmax(prediction_proba))
    confidence = float(np.max(prediction_proba))

    return prediction, confidence