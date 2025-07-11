import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

"""
Módulo de pré-processamento de dados para o pipeline de correspondência de candidatos a vagas
Etapas: 
 _ Limpeza dos dados
 _ Tratamento de valores ausentes
 _ Codificação de variáveis categóricas
 _ Vetorização de texto (TF-IDF)

Funções:
 - limpar_dados: Preenche valores ausentes nos DataFrames de candidatos e vagas.
 - construir_dataframe_features: Constrói um DataFrame de recursos a partir dos dados de correspondência, candidatos e vagas.
 - get_preprocessor: Cria um pré-processador que aplica TF-IDF para texto e OneHotEncoder para variáveis categóricas.
"""

def limpar_dados(applicant_df, job_df):
    logger.info("Limpando dados ausentes nos dataframes de candidatos e vagas")
    applicant_df.fillna({'formacao_e_idiomas': {}, 'informacoes_profissionais': {}}, inplace=True)
    job_df.fillna({'perfil_vaga': {}}, inplace=True)
    return applicant_df, job_df


def construir_dataframe_features(match_df, applicants, jobs):
    logger.info("Construindo dataframe de features a partir de candidatos e vagas")
    features, labels = [], []
    for _, row in match_df.iterrows():
        app = applicants.get(row['applicant_id'])
        job = jobs.get(row['job_id'])
        if not app or not job:
            continue
        # Construção das features
        features.append({...})
        labels.append(row['label'])

    logger.info(f"Total de pares candidato-vaga processados: {len(features)}")
    return pd.DataFrame(features), labels

def get_preprocessor(text_max_features=500):
    logger.info("Criando pipeline de pré-processamento (TF-IDF + OneHotEncoder)")
    text_features = ['cv', 'descricao_vaga']
    categorical_features = ['nivel_academico', 'nivel_ingles', 'nivel_espanhol', 'area_atuacao']

    preprocessor = ColumnTransformer([
        ('text_cv', TfidfVectorizer(max_features=text_max_features), 'cv'),
        ('text_desc', TfidfVectorizer(max_features=text_max_features), 'descricao_vaga'),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    return preprocessor
