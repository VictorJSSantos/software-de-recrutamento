from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
import os

# Configuração básica do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()
model = None

class InputData(BaseModel):
    nivel_academico: str
    nivel_ingles: str
    nivel_espanhol: str
    area_atuacao: str
    cv: str
    descricao_vaga: str

@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "model/model.joblib")
    try:
        model = joblib.load(model_path)
        logging.info(f"Modelo carregado com sucesso de: {model_path}")
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo: {e}")

@app.post("/predict")
def predict(data: InputData):
    logging.info("Requisição recebida para /predict")

    input_df = data.model_dump()
    input_df = {k: [v] for k, v in input_df.items()}
    
    try:
        prediction = model.predict(input_df)
        logging.info(f"Predição feita com sucesso: {prediction}")
        return {"resultado": int(prediction[0])}
    except Exception as e:
        logging.error(f"Erro durante a predição: {e}")
        return {"erro": "Erro ao realizar a predição"}