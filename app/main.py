# app/main.py
import logging
from pipeline import preprocessing  # exemplo de import

logging.basicConfig(
    filename='logs/preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# exemplo de chamada
preprocessing.executar_preprocessamento()
