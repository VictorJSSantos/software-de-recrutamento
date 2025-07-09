import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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
    """Adiciona uma coluna com a similaridade entre o CV e a descrição da vaga."""
    vectorizer = TfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(features_df['cv'] + ' ' + features_df['descricao_vaga'])
    sims = cosine_similarity(tfidf_matrix)
    # diagonal da similaridade entre o mesmo índice de CV e vaga
    features_df['similaridade_cv_vaga'] = [sims[i, i] for i in range(len(features_df))]
    return features_df

def adicionar_experiencia_previa(features_df, applicant_df):
    """Simples proxy: checa se a área de atuação já bate com a vaga"""
    def check_match(row):
        return 1 if row['area_atuacao'].lower() in row['descricao_vaga'].lower() else 0

    features_df['experiencia_na_area'] = features_df.apply(check_match, axis=1)
    return features_df

# Futuras features
def placeholder_features(features_df):
    # Tempo médio de resposta, participação em entrevistas, etc.
    features_df['tempo_medio_resposta'] = np.random.rand(len(features_df))  # Placeholder
    features_df['participacoes_anteriores'] = np.random.randint(0, 5, size=len(features_df))
    return features_df
