import json
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Imports internos do projeto
from pipeline.preprocessing import (
    limpar_dados,
    construir_dataframe_features,
    get_preprocessor
)
from pipeline.feature_engineering import (
    calcular_similaridade_cv_descricao,
    adicionar_experiencia_previa,
    placeholder_features
)

def log(title):
    print("\n" + "*" * 60)
    print(f"{title}".center(60))
    print("*" * 60 + "\n")

if __name__ == "__main__":
    start_time = time.time()
    log("INICIANDO PIPELINE")

    # 1. Carregamento dos dados
    log("CARREGANDO DADOS")
    with open('data/applicants.json', encoding='utf-8') as f:
        applicants = json.load(f)

    with open('data/vagas.json', encoding='utf-8') as f:
        jobs = json.load(f)

    with open('data/prospects.json', encoding='utf-8') as f:
        prospects = json.load(f)

    print("Dados carregados com sucesso.")

    # 2. Normalização dos dados base
    log("NORMALIZANDO")
    applicant_df = pd.DataFrame.from_dict(applicants, orient='index')
    job_df = pd.DataFrame.from_dict(jobs, orient='index')
    applicant_df, job_df = limpar_dados(applicant_df, job_df)

    # 3. Geração de labels e features
    log("GERANDO FEATURES E LABELS")
    rows = []
    for job_id, data in prospects.items():
        for p in data['prospects']:
            label = 1 if 'Contratado' in p['situacao_candidado'] else 0
            if p['codigo'] in applicants:
                rows.append({
                    'job_id': job_id,
                    'applicant_id': p['codigo'],
                    'label': label
                })
    match_df = pd.DataFrame(rows)

    features_df, labels = construir_dataframe_features(match_df, applicants, jobs)
    print(f"Total de registros: {len(features_df)}")

    # 4. Feature Engineering
    log("APLICANDO FEATURE ENGINEERING")
    features_df = calcular_similaridade_cv_descricao(features_df)
    features_df = adicionar_experiencia_previa(features_df, applicant_df)
    features_df = placeholder_features(features_df)

    # 5. Pré-processamento
    log("PREPARANDO PIPELINE")
    preprocessor = get_preprocessor()
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 6. Split & Treinamento
    log("TREINANDO MODELO")
    X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    # 7. Avaliação
    log("AVALIANDO MODELO")
    y_pred = clf.predict(X_test)
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # 8. Salvamento
    log("SALVANDO MODELO")
    joblib.dump(clf, 'model/model.joblib')
    print("Modelo salvo com sucesso.")

    log("PIPELINE FINALIZADO")
    print(f"Tempo total: {round(time.time() - start_time, 2)} segundos")