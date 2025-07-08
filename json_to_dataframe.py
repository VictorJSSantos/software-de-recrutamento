import pandas as pd
import json
import os


def achatar_dicionario(d, prefixo=""):
    item_achatado = {}
    for chave, valor in d.items():
        nova_chave = f"{prefixo}_{chave}" if prefixo else chave
        if isinstance(valor, dict):
            item_achatado.update(achatar_dicionario(valor, nova_chave))
        elif isinstance(valor, list):
            item_achatado[nova_chave] = str(valor)
        else:
            item_achatado[nova_chave] = valor
    return item_achatado


def json_para_dataframe_e_csv(caminho_json, caminho_csv_saida):
    try:
        with open(caminho_json, "r", encoding="utf-8") as f:
            dados_json = json.load(f)
    except FileNotFoundError:
        print(f"Erro: O arquivo JSON '{caminho_json}' não foi encontrado.")
        return
    except json.JSONDecodeError:
        print(
            f"Erro: Não foi possível decodificar o arquivo JSON '{caminho_json}'. Verifique a formatação."
        )
        return
    except Exception as e:
        print(f"Erro inesperado ao carregar o JSON: {e}")
        return

    if not isinstance(dados_json, dict):
        print(
            f"Erro: O conteúdo do arquivo JSON '{caminho_json}' não é um dicionário no nível raiz."
        )
        print(f"Tipo encontrado: {type(dados_json)}. Esperado: dict.")
        print("Verifique se o arquivo JSON está formatado como 'id: { ... }'.")
        return

    registros = []

    for id_principal, valores_colunas in dados_json.items():
        linha_atual = {"id": id_principal}

        if isinstance(valores_colunas, dict):
            linha_atual.update(achatar_dicionario(valores_colunas))
        else:
            linha_atual["valor"] = valores_colunas

        registros.append(linha_atual)

    df = pd.DataFrame(registros)

    try:
        df.to_csv(caminho_csv_saida, index=False, encoding="utf-8")
        print(f"Dados exportados com sucesso para '{caminho_csv_saida}'.")
    except Exception as e:
        print(f"Erro ao exportar para CSV: {e}")


if __name__ == "__main__":
    # --- Configurações para o seu arquivo JSON ---
    # Defina o caminho para o seu arquivo JSON de entrada
    # Exemplo: 'C:/Users/SeuUsuario/Documents/MeusDados/vagas.json'
    # Ou se estiver na mesma pasta do script: 'vagas.json'
    file = "prospects"
    meu_arquivo_json_entrada = (
        f"./data/{file}.json"  # <--- AJUSTE AQUI PARA O SEU ARQUIVO
    )

    # Defina o diretório onde o arquivo CSV de saída será salvo
    diretorio_saida = "./data"  # Opcional: mude para o diretório que preferir

    # Garante que o diretório de saída exista
    os.makedirs(diretorio_saida, exist_ok=True)

    # Nome do arquivo CSV de saída
    meu_arquivo_csv_saida = os.path.join(diretorio_saida, f"{file}.csv")

    # --- Chamada principal para processar o JSON ---
    json_para_dataframe_e_csv(meu_arquivo_json_entrada, meu_arquivo_csv_saida)

    # --- Verificação (Opcional) ---
    print("\n--- Conteúdo do CSV gerado (apenas para verificação) ---")
    try:
        df_verificacao = pd.read_csv(meu_arquivo_csv_saida)
        print(df_verificacao.head())
        print("\n--- Colunas do CSV gerado ---")
        print(df_verificacao.columns)
    except FileNotFoundError:
        print(f"Arquivo CSV '{meu_arquivo_csv_saida}' não encontrado para verificação.")
    except Exception as e:
        print(f"Erro ao ler o CSV para verificação: {e}")
