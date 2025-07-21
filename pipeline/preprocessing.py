import logging
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from unidecode import unidecode

# Função de limpeza de texto
def limpar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = unidecode(str(texto).lower())
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

# Geração de embedding TF-IDF com prefixo padronizado
def gerar_embedding(df, coluna_texto, max_features=300, prefixo="embedding"):
    texto_limpo = df[coluna_texto].fillna('').apply(limpar_texto)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texto_limpo)
    return pd.DataFrame(X.toarray(), columns=[f'{prefixo}_{i}' for i in range(X.shape[1])])

# Função principal de pré-processamento (tipo='applicant' ou 'vaga')
def preprocessar_dados(df, tipo='applicant', logger=None):
    logger = logger or logging.getLogger(__name__)
    logger.info(f"/pipeline/preprocessing.py - Iniciando pré-processamento dos dados ({tipo})")
    df = df.copy()

    # Colunas textuais comuns para limpeza
    colunas_texto = [
        "descricao_vaga", "tecnologias_desejadas", "formacao", 
        "entrevistador", "respostas", "cv_pt", "cv_en"
    ]
    for col in colunas_texto:
        if col in df.columns:
            df[col] = df[col].apply(limpar_texto)
            logger.info(f"/pipeline/preprocessing.py - Texto limpo na coluna: {col}")

    # Feature numérica: número de palavras em 'respostas'
    if "respostas" in df.columns:
        df["respostas_num_palavras"] = df["respostas"].apply(lambda x: len(str(x).split()))
    else:
        df["respostas_num_palavras"] = 0

    # LabelEncoder para 'nivel' (ajuste para prefixo 'perfil_vaga.nivel' no tipo vaga)
    nivel_col = "nivel" if tipo == "applicant" else "perfil_vaga.nivel"
    if nivel_col in df.columns:
        le = LabelEncoder()
        df["nivel_encoded"] = le.fit_transform(df[nivel_col].astype(str))
        logger.info(f"/pipeline/preprocessing.py - LabelEncoder aplicado na coluna '{nivel_col}' ({tipo}).")

    # Texto unificado por tipo
    df["texto_unificado"] = ""
    if tipo == 'applicant':
        colunas_unificadas = [
            "cv_pt", "informacoes_profissionais.titulo_profissional",
            "informacoes_profissionais.area_atuacao", "informacoes_profissionais.conhecimentos_tecnicos",
            "formacao_e_idiomas.nivel_academico"
        ]
    elif tipo == 'vaga':
        colunas_unificadas = [
            "infos_basicas.titulo_vaga",
            "perfil_vaga.principais_atividades",
            "perfil_vaga.competencia_tecnicas_e_comportamentais",
            "perfil_vaga.nivel_profissional",
            "perfil_vaga.nivel_academico"
        ]
    else:
        logger.warning(f"Tipo '{tipo}' desconhecido.")
        colunas_unificadas = []

    for col in colunas_unificadas:
        if col in df.columns:
            df["texto_unificado"] += " " + df[col].fillna("").astype(str)

    df["texto_unificado"] = df["texto_unificado"].apply(limpar_texto)

    # Embeddings
    try:
        embeddings = gerar_embedding(df, "texto_unificado", prefixo="embedding")
        df = pd.concat([df.reset_index(drop=True), embeddings.reset_index(drop=True)], axis=1)
        logger.info(f"/pipeline/preprocessing.py - Embeddings TF-IDF gerados com sucesso para {tipo}.")
    except Exception as e:
        logger.error(f"/pipeline/preprocessing.py - Erro ao gerar embeddings ({tipo}): {e}")

    df.fillna("", inplace=True)
    logger.info(f"/pipeline/preprocessing.py - Pré-processamento concluído para {tipo}.")
    return df


    
# Pré-processa uma única string de entrada (como a descrição de candidato ou vaga).
def preprocess_text(texto: str) -> str:
    
    
    return limpar_texto(texto)