from fastapi import FastAPI, HTTPException
from app.schema import MatchRequest, MatchResponse
from pipeline.preprocessing import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import datetime


# Inicializa app
app = FastAPI()

# Configura logging (opcional mas recomendado)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict", response_model=MatchResponse)
def predict(data: MatchRequest):
    try:
        # Verificar campos vazios
        if not data.descricao_candidato.strip() or not data.descricao_vaga.strip():
            raise HTTPException(
                status_code=400,
                detail="Os campos 'descricao_candidato' e 'descricao_vaga' não podem estar vazios."
            )

        # Pré-processar os textos
        candidato = preprocess_text(data.descricao_candidato)
        vaga = preprocess_text(data.descricao_vaga)

        # Vetorização com TF-IDF
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([candidato, vaga])

        # Similaridade cosseno
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        match = int(sim > 0.5)  # Limiar pode ser ajustado

        #No retorno da API
        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "candidato": data.descricao_candidato[:100],
            "vaga": data.descricao_vaga[:100],
            "similaridade": round(sim, 2),
            "match": match
        }

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(f"logs/06_predictions_log_{timestamp_str}.csv", "a", encoding="utf-8") as f:
            f.write(f"{log_data['timestamp']},{log_data['similaridade']},{log_data['match']}\n")


        return MatchResponse(match=match, similaridade=round(sim, 2))

    except HTTPException as http_err:
        # Relevanta a exceção esperada para o FastAPI lidar
        raise http_err

    except Exception as e:
        logger.exception("Erro interno durante a execução do endpoint /predict.")
        raise HTTPException(
            status_code=500,
            detail="Erro interno na API. Verifique os logs para mais detalhes."
        )
