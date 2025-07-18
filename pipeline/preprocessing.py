import logging
from sklearn.preprocessing import LabelEncoder
from .utils import limpar_texto

def preprocessar_dados(df):
    logging.info("Iniciando pré-processamento dos dados")

    colunas_texto = ["descricao_vaga", "tecnologias_desejadas", "formacao", "entrevistador", "respostas", "cv_pt", "cv_en"]
    for col in colunas_texto:
        if col in df.columns:
            df[col] = df[col].apply(limpar_texto)
            logging.info(f"Texto limpo na coluna: {col}")
        else:
            logging.debug(f"Coluna {col} ausente no dataset.")

    if "nivel" in df.columns:
        le = LabelEncoder()
        df["nivel_encoded"] = le.fit_transform(df["nivel"].astype(str))
        logging.info("Codificação LabelEncoder aplicada na coluna 'nivel'")
    else:
        logging.debug("Coluna 'nivel' não encontrada. Codificação ignorada.")

    if "respostas" in df.columns:
        df["respostas_num_palavras"] = df["respostas"].apply(lambda x: len(str(x).split()))
        logging.info("Feature 'respostas_num_palavras' criada.")
    else:
        df["respostas_num_palavras"] = 0

    df.fillna("", inplace=True)
    return df
