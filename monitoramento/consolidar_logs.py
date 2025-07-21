import pandas as pd
import glob
import os

# Diretório onde estão os logs individuais
input_dir = "./logs/"
padrao_arquivos = os.path.join(input_dir, "06_predictions_log_*.csv")
arquivos_log = glob.glob(padrao_arquivos)

# Diretório de saída do consolidado
output_dir = "./monitoramento/logs/"
os.makedirs(output_dir, exist_ok=True)

# Lista para armazenar os DataFrames
dfs = []

for arquivo in arquivos_log:
    try:
        # Define o cabeçalho manualmente
        df = pd.read_csv(arquivo, names=["timestamp", "similaridade", "match"], header=None)

        # Verifica se os dados têm o tipo esperado
        if df.shape[1] == 3:
            dfs.append(df)
        else:
            print(f"Formato inválido em {arquivo}: {df.shape[1]} colunas")
    except Exception as e:
        print(f"Erro ao ler {arquivo}: {e}")

# Concatena os DataFrames
if dfs:
    df_consolidado = pd.concat(dfs).drop_duplicates()
    df_consolidado = df_consolidado.sort_values(by="timestamp")
    df_consolidado.to_csv(os.path.join(output_dir, "06_predictions_log.csv"), index=False)
    print("** Logs consolidados com sucesso.")
else:
    print("** Nenhum log encontrado para consolidar.")