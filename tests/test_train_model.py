import pandas as pd
from models.train_model import treinar_modelo

# Carrega o dataset com a variável 'match'
df = pd.read_csv("data/processed/dataset_final.csv")

# Treina o modelo
treinar_modelo(df)

print("Treinamento finalizado.")
