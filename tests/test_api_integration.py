from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_full_pipeline():
    payload = {
        "descricao_candidato": "Desenvolvedora backend com experiência em Python, Flask e testes automatizados.",
        "descricao_vaga": "Experiência com Django e APIs REST"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert "match" in result
    assert "similaridade" in result

    assert isinstance(result["match"], int)
    assert result["match"] in [0, 1]

    assert isinstance(result["similaridade"], float)
    assert 0.0 <= result["similaridade"] <= 1.0
