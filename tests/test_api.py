from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_success():
    payload = {
        "features": [0.1, 0.2, 0.3]  # 3 features conforme esperado pelo modelo
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "probability" in json_response
    assert isinstance(json_response["probability"], float)
    assert 0.0 <= json_response["probability"] <= 1.0


def test_predict_invalid_input():
    # Input inválido: string ao invés de lista de floats
    payload = {"features": "invalid"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # erro de validação
