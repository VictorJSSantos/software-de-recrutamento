import os
import logging
from datetime import datetime

from pipeline.loaders import carregar_applicants, carregar_vagas, carregar_prospects
from pipeline.preprocessing import preprocessar_dados
from pipeline.dataset_builder import build_dataset
from models.train_model import treinar_modelo


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


def main():
    # --- Etapa 1: Carregar dados brutos ---
    logger_load = configurar_logger("01_carregamento")
    logger_load.info("run_pipeline.py - Iniciando pipeline de carregamento de dados brutos")

    df_applicants = carregar_applicants()
    logger_load.info(f"run_pipeline.py - applicants.json carregado com {len(df_applicants)} registros.")

    df_vagas = carregar_vagas()
    logger_load.info(f"run_pipeline.py - vagas.json carregado com {len(df_vagas)} registros.")

    df_prospects = carregar_prospects()
    logger_load.info(f"run_pipeline.py - prospects.json carregado com {len(df_prospects)} registros.")

    # --- Etapa 2: Pré-processamento ---
    logger_pre = configurar_logger("02_preprocessamento")
    logger_pre.info("run_pipeline.py - Iniciando etapa de pré-processamento")

    df_applicants = preprocessar_dados(df_applicants, tipo='applicant', logger=logger_pre)
    df_vagas = preprocessar_dados(df_vagas, tipo='vaga', logger=logger_pre)
    df_prospects = preprocessar_dados(df_prospects, tipo='prospect', logger=logger_pre)

    logger_pre.info("run_pipeline.py - Pré-processamento concluído para todos os dados.")

    # --- Etapa 3: Gerar dataset final com match ---
    logger_ds = configurar_logger("03_geracao_dataset")
    logger_ds.info("run_pipeline.py - Iniciando geração do dataset final")

    output_path = "data/processed/dataset_final.csv"
    df_dataset = build_dataset(df_applicants, df_vagas, df_prospects, output_path=output_path, logger=logger_ds)

    logger_ds.info(f"run_pipeline.py - Dataset final criado com {len(df_dataset)} registros.")

    # --- Etapa 4: Treinamento do modelo ---
    logger_train = configurar_logger("04_treinamento_modelo")
    logger_train.info("run_pipeline.py - Iniciando treinamento do modelo preditivo")

    treinar_modelo(df_dataset, logger=logger_train)

    logger_train.info("run_pipeline.py - Treinamento finalizado com sucesso")


if __name__ == "__main__":
    # Configura o logger raiz para INFO — assim qualquer log sem logger explícito tb aparece no console
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
