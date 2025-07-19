import os
import logging
import joblib
import pandas as pd
from datetime import datetime
from time import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Caminhos
MODEL_PATH = "models/modelo_match.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
DATA_PATH = "data/processed/dataset_final.csv"

# Log com timestamp
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_filename = f"train_model_{timestamp}.log"
log_path = os.path.join(log_dir, log_filename)

# Logging terminal + arquivo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def treinar_modelo(df: pd.DataFrame = None, return_model: bool = False):
    start_time = time()
    logging.info("Iniciando pipeline de treinamento...")

    if df is None:
        logging.info(f"Lendo dataset: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)

    if 'descricao_completa' not in df.columns:
        logging.error("Coluna 'descricao_completa' não encontrada no dataset.")
        raise ValueError("Coluna 'descricao_completa' não encontrada no dataset.")

    X_text = df["descricao_completa"].fillna("")
    y = df["match"]

    logging.info("Vetorizando com TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(tqdm(X_text, desc="TF-IDF"))

    logging.info("Dividindo em treino/teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logging.info("Buscando melhores hiperparâmetros (RandomizedSearchCV)...")
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="roc_auc",
        n_jobs=-1,
        cv=3,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    logging.info(f"Melhor modelo encontrado: {search.best_params_}")

    logging.info("Avaliando modelo no conjunto de teste...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    logging.info(f"Acurácia: {acc:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    logging.info(f"Modelo salvo em: {MODEL_PATH}")
    logging.info(f"Vetorizador salvo em: {VECTORIZER_PATH}")

    tempo_total = time() - start_time
    logging.info(f"Tempo total de execução: {tempo_total:.2f} segundos")
    logging.info(f"Log salvo em: {log_path}")

    if return_model:
        return best_model, vectorizer, {"accuracy": acc, "roc_auc": roc_auc}
