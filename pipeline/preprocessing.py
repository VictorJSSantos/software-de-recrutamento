import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

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
    # Preenche valores ausentes
    applicant_df.fillna({'formacao_e_idiomas': {}, 'informacoes_profissionais': {}}, inplace=True)
    job_df.fillna({'perfil_vaga': {}}, inplace=True)
    return applicant_df, job_df

def construir_dataframe_features(match_df, applicants, jobs):
    features, labels = [], []

    for _, row in match_df.iterrows():
        app = applicants.get(row['applicant_id'])
        job = jobs.get(row['job_id'])
        if not app or not job:
            continue

        features.append({
            'nivel_academico': app['formacao_e_idiomas'].get('nivel_academico', 'Desconhecido'),
            'nivel_ingles': app['formacao_e_idiomas'].get('nivel_ingles', 'Desconhecido'),
            'nivel_espanhol': app['formacao_e_idiomas'].get('nivel_espanhol', 'Desconhecido'),
            'area_atuacao': app['informacoes_profissionais'].get('area_atuacao', 'Desconhecido'),
            'cv': app.get('cv_pt', '')[:3000],
            'descricao_vaga': job['perfil_vaga'].get('principais_atividades', '') + '\n' + job['perfil_vaga'].get('competencia_tecnicas_e_comportamentais', '')
        })
        labels.append(row['label'])

    return pd.DataFrame(features), labels

def get_preprocessor(text_max_features=500):
    text_features = ['cv', 'descricao_vaga']
    categorical_features = ['nivel_academico', 'nivel_ingles', 'nivel_espanhol', 'area_atuacao']

    preprocessor = ColumnTransformer([
        ('text_cv', TfidfVectorizer(max_features=text_max_features), 'cv'),
        ('text_desc', TfidfVectorizer(max_features=text_max_features), 'descricao_vaga'),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    return preprocessor
