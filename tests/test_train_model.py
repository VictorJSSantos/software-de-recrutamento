import pytest
import pandas as pd
import os
from models import train_model

import logging
from datetime import datetime



def configurar_logger(nome_etapa):
    """Configura o logger para a etapa."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{nome_etapa}_{timestamp}.log")

    logger = logging.getLogger(nome_etapa)
    logger.setLevel(logging.INFO)

    # Remove handlers antigos para evitar logs duplicados
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger



# Configura o logger raiz para INFO — assim qualquer log sem logger explícito tb aparece no console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
   

def test_treinar_modelo_com_dataset_real():

        logger_train = configurar_logger("teste_treinamento_modelo")
        logger_train.info("tests/run_test_train_model.py - Iniciando teste do treinamento do modelo preditivo")


        logger_train.info("tests/run_test_train_model.py - Testa se o modelo treina com o dataset real sem erros e salva o arquivo.")
        df = pd.read_csv("data/processed/dataset_final.csv")


        logger_train.info("tests/run_test_train_model.py -Garante que o modelo ainda não existe antes do teste")
        # Garante que o modelo ainda não existe antes do teste
        caminho_modelo = "models/modelo_match.pkl"
        if os.path.exists(caminho_modelo):
            os.remove(caminho_modelo)

        logger_train.info("tests/run_test_train_model.py - Chama o treinamento do modelo preditivo")
        train_model.treinar_modelo(df, logger=logger_train)

        logger_train.info("tests/run_test_train_model.py - Treinamento finalizado com sucesso")


        # Valida se o modelo foi salvo com sucesso
        logger_train.info("tests/run_test_train_model.py - Valida se o modelo foi salvo com sucesso")

        assert os.path.exists(caminho_modelo), "Modelo não foi salvo após o treinamento."
