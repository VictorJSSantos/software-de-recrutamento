import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from pipeline import preprocessing, feature_engineering
import os
import time

from utils.logger import get_logger
logger = get_logger("train_model")


def log(title):
    print("\n" + "*"*60)
    print(f"{title}".center(60))
    print("*"*60 + "\n")

def carregar_dados():
    with open('data/applicants.json', encoding='utf-8') as f:
        applicants = json.load(f)
    with open('data/vagas.json', encoding='utf-8') as f:
        jobs = json.load(f)
    with open('data/prospects.json', encoding='utf-8') as f:
        prospects = json.load(f)
    return applicants, jobs, prospects

def construir_match_dataframe(prospects):
    rows = []
    for job_id, data in prospects.items():
        for p in data['prospects']:
            label = 1 if 'Contratado' in p['situacao_candidado'] else 0
            rows.append({
                'job_id': job_id,
                'applicant_id': p['codigo'],
                'label': label
            })
    return pd.DataFrame(rows)

def main():
    start_time = time.time()

    log("INICIANDO TREINAMENTO DO MODELO")
    logger.info("Iniciando pipeline de treinamento")

    # Etapa 1: Carregar os dados
    log("CARREGANDO DADOS")
    applicants, jobs, prospects = carregar_dados()

    # Etapa 2: Normalizar os dados
    log("NORMALIZANDO DADOS")
    applicant_df = pd.DataFrame.from_dict(applicants, orient='index')
    job_df = pd.DataFrame.from_dict(jobs, orient='index')
    applicant_df, job_df = preprocessing.limpar_dados(applicant_df, job_df)

    # Etapa 3: Construir DataFrame de match
    log("CRIANDO DATAFRAME DE MATCH")
    match_df = construir_match_dataframe(prospects)

    # Etapa 4: Feature Engineering
    log("CONSTRUINDO FEATURES")
    features_df, labels = preprocessing.construir_dataframe_features(match_df, applicants, jobs)

    features_df = feature_engineering.calcular_similaridade_cv_descricao(features_df)
    features_df = feature_engineering.adicionar_experiencia_previa(features_df, applicant_df)
    features_df = feature_engineering.placeholder_features(features_df)

    # Etapa 5: Pipeline de Treinamento
    log("PREPARANDO PIPELINE")
    preprocessor = preprocessing.get_preprocessor(text_max_features=500)
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Etapa 6: Treinamento
    log("TREINANDO MODELO")
    X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    # Etapa 7: Salvar modelo
    log("SALVANDO MODELO")
    logger.info("Salvando modelo treinado")

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/model.joblib")

    log("TREINAMENTO FINALIZADO")
    logger.info("Treinamento finalizado com sucesso")
    print(f"Tempo total: {round(time.time() - start_time, 2)} segundos")

if __name__ == "__main__":
    main()