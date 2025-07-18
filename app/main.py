
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schema import MatchInput, MatchOutput
from app.model import carregar_modelo, prever
from pipeline import preprocessing

# Configura logging para pipeline
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Inicia FastAPI
app = FastAPI(
    title="Decision AI Recruiter",
    description="API para previsão de match entre candidatos e vagas com ML",
    version="1.0.0"
)

# Executa pipeline de pré-processamento (ex: se necessário rodar ao iniciar)
# Descomente se desejar sempre executar ao rodar a API
# preprocessing.executar_preprocessamento()

# Carrega o modelo uma vez ao iniciar a API
modelo = carregar_modelo()

@app.get("/")
def read_root():
    return {"message": "API da Decision AI Recruiter online com sucesso!"}

@app.post("/predict", response_model=MatchOutput)
def predict_match(data: MatchInput):
    try:
        # Recebe os dados de entrada como lista de features
        df = pd.DataFrame([data.features])

        # Faz a predição
        pred, prob = prever(modelo, df)

        return MatchOutput(prediction=pred[0], probability=prob[0])

    except Exception as e:
        logging.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao prever: {str(e)}")
