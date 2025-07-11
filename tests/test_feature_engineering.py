import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
from pipeline import feature_engineering


def test_calcular_similaridade_cv_descricao():
    df = pd.DataFrame({
        'cv': ['Experiência em análise de dados.'],
        'descricao_vaga': ['Buscamos alguém com experiência em dados.']
    })
    df_result = feature_engineering.calcular_similaridade_cv_descricao(df)
    assert 'similaridade_cv_vaga' in df_result.columns
    assert 0 <= df_result['similaridade_cv_vaga'][0] <= 1

def test_adicionar_experiencia_previa():
    df = pd.DataFrame({
        'area_atuacao': ['TI'],
        'descricao_vaga': ['Vaga para atuação em TI e suporte']
    })
    df_result = feature_engineering.adicionar_experiencia_previa(df, pd.DataFrame())
    assert 'experiencia_na_area' in df_result.columns
    assert df_result['experiencia_na_area'].iloc[0] == 1

def test_placeholder_features():
    df = pd.DataFrame({'cv': ['abc'], 'descricao_vaga': ['def']})
    df_result = feature_engineering.placeholder_features(df)
    assert 'tempo_medio_resposta' in df_result.columns
    assert 'participacoes_anteriores' in df_result.columns
    assert 0 <= df_result['tempo_medio_resposta'].iloc[0] <= 1
    assert 0 <= df_result['participacoes_anteriores'].iloc[0] < 5
