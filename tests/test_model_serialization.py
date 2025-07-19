import os
import joblib
from sklearn.base import ClassifierMixin

def test_model_persistence():
    assert os.path.exists("models/modelo_match.pkl"), "Modelo não foi salvo"
    modelo = joblib.load("models/modelo_match.pkl")
    assert isinstance(modelo, ClassifierMixin), "Modelo carregado é inválido"
