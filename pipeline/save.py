import os
import logging

PROCESSED_PATH = "data/processed"

def salvar_df(df, nome_arquivo):
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    caminho = os.path.join(PROCESSED_PATH, nome_arquivo)
    df.to_csv(caminho, index=False)
    logging.info(f"Dados salvos em '{caminho}'")
