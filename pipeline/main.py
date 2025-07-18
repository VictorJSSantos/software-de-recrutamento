from datetime import datetime
import logging

from .utils import limpar_diretorio, configurar_logging
from .loaders import carregar_prospects, carregar_applicants, carregar_vagas
from .preprocessing import preprocessar_dados
from .save import salvar_df

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = f"logs/preprocessing_{timestamp}.log"
    configurar_logging(log_path)

    try:
        logging.info("Limpando dados processados antigos...")
        limpar_diretorio("data/processed")

        logging.info("Carregando dados brutos...")
        df_prospects = carregar_prospects()
        df_applicants = carregar_applicants()
        df_vagas = carregar_vagas()

        logging.info("Pré-processando dados...")
        df_prospects = preprocessar_dados(df_prospects)
        df_applicants = preprocessar_dados(df_applicants)
        df_vagas = preprocessar_dados(df_vagas)

        logging.info("Salvando dados processados...")
        salvar_df(df_prospects, "prospects_processed.csv")
        salvar_df(df_applicants, "applicants_processed.csv")
        salvar_df(df_vagas, "vagas_processed.csv")

        logging.info("Pipeline concluída com sucesso!")

    except Exception as e:
        logging.critical(f"Erro fatal na execução: {e}", exc_info=True)

if __name__ == "__main__":
    main()
