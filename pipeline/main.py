import os
import logging
from datetime import datetime

from .utils import limpar_diretorio, configurar_logging
from .loaders import carregar_prospects, carregar_applicants, carregar_vagas
from .preprocessing import preprocessar_dados
from .save import salvar_df

def garantir_diretorios():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def main():
    garantir_diretorios()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = f"logs/preprocessing_{timestamp}.log"
    configurar_logging(log_path)

    try:
        logging.info("Início do pipeline de pré-processamento")

        # Limpa arquivos processados antigos
        logging.info("Limpando diretório data/processed")
        limpar_diretorio("data/processed")

        # Carrega os dados
        logging.info("Carregando dados brutos...")
        df_prospects = carregar_prospects()
        df_applicants = carregar_applicants()
        df_vagas = carregar_vagas()

        # Verifica se foram carregados corretamente
        for nome, df in zip(
            ["prospects", "applicants", "vagas"],
            [df_prospects, df_applicants, df_vagas]
        ):
            if df.empty:
                logging.warning(f"Atenção: O DataFrame '{nome}' está vazio.")

        # Pré-processa os dados
        logging.info("Aplicando pré-processamento...")
        df_prospects = preprocessar_dados(df_prospects)
        df_applicants = preprocessar_dados(df_applicants, tipo='applicant')
        df_vagas = preprocessar_dados(df_vagas, tipo='vaga')

        # Salva os resultados
        logging.info("Salvando arquivos processados em data/processed/")
        salvar_df(df_prospects, "prospects_processed.csv")
        salvar_df(df_applicants, "applicants_processed.csv")
        salvar_df(df_vagas, "vagas_processed.csv")

        logging.info("Pipeline concluída com sucesso!")

    except Exception as e:
        logging.critical(f"Erro fatal na execução: {e}", exc_info=True)

if __name__ == "__main__":
    main()
