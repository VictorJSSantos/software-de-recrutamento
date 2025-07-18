import pandas as pd
import pytest
from models.train_model import treinar_modelo
from sklearn.base import ClassifierMixin

@pytest.fixture
def dataset():
    df = pd.read_csv("data/processed/dataset_final.csv")
    return df

@pytest.mark.parametrize("min_acc, min_auc", [(0.75, 0.85)])
def test_model_performance(dataset, min_acc, min_auc):
    modelo, metrics = treinar_modelo(dataset, return_model=True)

    # Verifica se retornou um classificador
    assert isinstance(modelo, ClassifierMixin), "Modelo não é um classificador válido"

    # Validação das métricas
    assert metrics["accuracy"] >= min_acc, f"Accuracy baixa: {metrics['accuracy']:.2f}"
    assert metrics["roc_auc"] >= min_auc, f"ROC-AUC baixo: {metrics['roc_auc']:.2f}"

def test_model_prediction_shape(dataset):
    modelo, _ = treinar_modelo(dataset, return_model=True)
    X = dataset.drop(columns=[col for col in ["match", "id_candidato", "id_vaga"] if col in dataset.columns])

    pred = modelo.predict(X.head(1))
    assert len(pred) == 1, "Predição não retornou um único resultado"
