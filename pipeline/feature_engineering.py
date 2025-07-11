import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.logger import get_logger

logger = get_logger(__name__)


"""
Módulo de engenharia de recursos para o pipeline de correspondência de candidatos a vagas
Etapas:
 _ Cálculo de similaridade entre CV e descrição da vaga
 _ Adição de experiência prévia na área de atuação
 _ Placeholder para futuras features
 
Funções:
 - calcular_similaridade_cv_descricao: Adiciona uma coluna com a similaridade entre o CV e a descrição da vaga.
 - adicionar_experiencia_previa: Adiciona uma coluna indicando se o candidato tem experiência prévia na área de atuação da vaga.
 - placeholder_features: Placeholder para futuras features como tempo médio de resposta, participações anteriores, etc.
"""

def calcular_similaridade_cv_descricao(features_df):
    logger.info("Calculando similaridade entre CV e descrição da vaga") 
    """Adiciona uma coluna com a similaridade entre o CV e a descrição da vaga, calculando linha a linha"""
    vectorizer = TfidfVectorizer(max_features=500)

    sims = []
    for cv, desc in zip(features_df['cv'], features_df['descricao_vaga']):
        if not cv.strip() or not desc.strip():
            sims.append(0.0)
            continue

        corpus = [cv, desc]

        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        except ValueError:
            similarity = 0.0

        sims.append(similarity)

    features_df['similaridade_cv_vaga'] = sims
    return features_df

def adicionar_experiencia_previa(features_df, applicant_df):
    logger.info("Adicionando flag de experiência prévia na área")
    """Simples proxy: checa se a área de atuação já bate com a vaga"""
    def check_match(row):
        return 1 if row['area_atuacao'].lower() in row['descricao_vaga'].lower() else 0

    features_df['experiencia_na_area'] = features_df.apply(check_match, axis=1)
    return features_df

# Futuras features
def placeholder_features(features_df):
    logger.info("Adicionando features placeholder (tempo de resposta, entrevistas...)")
    # Tempo médio de resposta, participação em entrevistas, etc.
    features_df['tempo_medio_resposta'] = np.random.rand(len(features_df))  # Placeholder
    features_df['participacoes_anteriores'] = np.random.randint(0, 5, size=len(features_df))
    return features_df
