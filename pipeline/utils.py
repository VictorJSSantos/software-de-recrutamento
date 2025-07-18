import os
import re
import logging
from unidecode import unidecode
import shutil

def limpar_texto(texto):
    if texto is None:
        return ""
    texto = unidecode(str(texto).lower())
    texto = re.sub(r"[^a-zA-Z0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def limpar_diretorio(caminho):
    if os.path.exists(caminho):
        shutil.rmtree(caminho)
    os.makedirs(caminho)

def configurar_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
