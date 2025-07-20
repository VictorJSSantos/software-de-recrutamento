import argparse
import logging
import os

from pipeline.main import main as preprocessar_dados
from tests.test_build_dataset import build_dataset
from models.train_model import treinar_modelo
import pandas as pd

# Configura logger simples
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def executar_pipeline_completa():
    logging.info(
        "Iniciando pipeline completa: preprocessamento -> geração de dataset -> treinamento"
    )

    # 1. Pré-processamento e geração dos embeddings
    logging.info("Etapa 1: Pré-processamento")
    preprocessar_dados()

    # 2. Geração do dataset final com a variável 'match'
    logging.info("Etapa 2: Geração do dataset final")
    build_dataset(
        applicants_path="data/processed/applicants_processed.csv",
        vagas_path="data/processed/vagas_processed.csv",
        prospect_path="data/processed/prospects_processed.csv",
        output_path="data/processed/dataset_final.csv",
        negatives_ratio=1,
    )

    # 3. Treinamento do modelo
    logging.info("Etapa 3: Treinamento do modelo")
    df = pd.read_csv("data/processed/dataset_final.csv")
    treinar_modelo(df)

    logging.info("Pipeline completa finalizada com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa todas as etapas da pipeline")
    parser.add_argument("--auto", action="store_true", help="Executar automaticamente")
    args = parser.parse_args()

    try:
        executar_pipeline_completa()
    except Exception as e:
        logging.error(f"Erro durante execução da pipeline: {e}")
