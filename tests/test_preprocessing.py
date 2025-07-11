import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import preprocessing
import pandas as pd



def test_limpar_dados():
    applicants = pd.DataFrame([{'formacao_e_idiomas': None, 'informacoes_profissionais': None}])
    jobs = pd.DataFrame([{'perfil_vaga': None}])

    a_clean, j_clean = preprocessing.limpar_dados(applicants.copy(), jobs.copy())

    assert isinstance(a_clean, pd.DataFrame)
    assert isinstance(j_clean, pd.DataFrame)
    assert isinstance(a_clean.loc[0, 'formacao_e_idiomas'], dict)
    assert isinstance(j_clean.loc[0, 'perfil_vaga'], dict)

def test_construir_dataframe_features():
    match_df = pd.DataFrame([{'job_id': 'j1', 'applicant_id': 'a1', 'label': 1}])
    applicants = {
        'a1': {
            'formacao_e_idiomas': {'nivel_academico': 'Superior', 'nivel_ingles': 'Avançado', 'nivel_espanhol': 'Básico'},
            'informacoes_profissionais': {'area_atuacao': 'TI'},
            'cv_pt': 'Experiência em Python, dados...'
        }
    }
    jobs = {
        'j1': {
            'perfil_vaga': {
                'principais_atividades': 'Desenvolver sistemas',
                'competencia_tecnicas_e_comportamentais': 'Trabalho em equipe'
            }
        }
    }

    features_df, labels = preprocessing.construir_dataframe_features(match_df, applicants, jobs)

    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 1
    assert features_df.iloc[0]['nivel_academico'] == 'Superior'
    assert isinstance(labels, list)
    assert labels == [1]

def test_get_preprocessor():
    pre = preprocessing.get_preprocessor()
    assert hasattr(pre, 'fit')
    assert hasattr(pre, 'transform')
