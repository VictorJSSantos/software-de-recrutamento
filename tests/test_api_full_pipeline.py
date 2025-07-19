from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_full_pipeline():
    payload = {
        "vaga": {
            "titulo": "Desenvolvedor Python",
            "descricao": "Experiência com Django e APIs REST"
        },
        "candidato": {
            "nome": "Ana Souza",
            "resumo": "Desenvolvedora backend com experiência em Python, Flask e testes automatizados."
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "probability" in result
    assert isinstance(result["probability"], float)
    assert 0 <= result["probability"] <= 1.0
