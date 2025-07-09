import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import time

def log(title):
    print("\n" + "*"*60)
    print(f"{title}".center(60))
    print("*"*60 + "\n\n")

start_time = time.time()
    
log("INICIANDO PIPELINE")

log("CARREGANDO DADOS")

with open('data/applicants.json', encoding='utf-8') as f:
    applicants = json.load(f)

with open('data/vagas.json', encoding='utf-8') as f:
    jobs = json.load(f)

with open('data/prospects.json', encoding='utf-8') as f:
    prospects = json.load(f)

print("Dados carregados com sucesso.")

log("NORMALIZANDO DADOS")
applicant_df = pd.DataFrame.from_dict(applicants, orient='index')
job_df = pd.DataFrame.from_dict(jobs, orient='index')
print(" Dados normalizados.")

log("PROCESSANDO MATCHS CANDIDATO-VAGA")
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
print(f" {len(match_df)} correspondências extraídas.")

log("CRIANDO FEATURES E LABELS")
features = []
labels = []

for _, row in match_df.iterrows():
    app = applicants.get(row['applicant_id'])
    job = jobs.get(row['job_id'])

    if not app or not job:
        continue

    feature = {
        'nivel_academico': app['formacao_e_idiomas']['nivel_academico'],
        'nivel_ingles': app['formacao_e_idiomas']['nivel_ingles'],
        'nivel_espanhol': app['formacao_e_idiomas']['nivel_espanhol'],
        'area_atuacao': app['informacoes_profissionais']['area_atuacao'],
        'titulo_profissional': app['informacoes_profissionais']['titulo_profissional'],
        'cv': app.get('cv_pt', '')[:3000],
        'descricao_vaga': job['perfil_vaga']['principais_atividades'] + '\n' + job['perfil_vaga']['competencia_tecnicas_e_comportamentais']
    }
    features.append(feature)
    labels.append(row['label'])

features_df = pd.DataFrame(features)
labels = np.array(labels)
print(f"Total de features criadas: {len(features_df)}")

log("PREPARANDO PIPELINE")
text_features = ['cv', 'descricao_vaga']
categorical_features = ['nivel_academico', 'nivel_ingles', 'nivel_espanhol', 'area_atuacao']

preprocessor = ColumnTransformer([
    ('text', TfidfVectorizer(max_features=500), 'cv'),
    ('text_job', TfidfVectorizer(max_features=500), 'descricao_vaga'),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

log("TREINANDO MODELO")
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
print("Modelo treinado com sucesso.")

log("AVALIAÇÃO")
y_pred = clf.predict(X_test)
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

log("SALVANDO MODELO")
joblib.dump(clf, 'model/model.joblib')
print("Modelo salvo em 'model/model.joblib'")

log("PIPELINE FINALIZADO")
print(f"Tempo total de execução: {round(time.time() - start_time, 2)} segundos")
