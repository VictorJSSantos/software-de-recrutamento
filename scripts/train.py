import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.train_model import treinar_modelo
import pandas as pd

if __name__ == "__main__":
    dataset_path = "../data/processed/dataset_final.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {dataset_path}")

    df = pd.read_csv(dataset_path)

    if df.empty:
        raise ValueError(
            "O dataset está vazio. Verifique a geração de dataset_final.csv."
        )

    treinar_modelo(df)
