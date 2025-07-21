import os
import time
import logging
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

# Caminhos
MODEL_PATH = "models/modelo_match.pkl"
DATASET_PATH = "data/processed/dataset_final.csv"
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)



def treinar_modelo(df: pd.DataFrame, return_model: bool = False, logger=None):
    logger.info("/models/train_model.py - Iniciando treinamento do modelo preditivo de match.")
    inicio = time.time()

    try:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Arquivo {DATASET_PATH} não encontrado.")

        df = pd.read_csv(DATASET_PATH)
        if df.empty:
            raise ValueError("Dataset vazio. Abortando treinamento.")

        if "match" not in df.columns or "similarity" not in df.columns:
            raise ValueError("Colunas obrigatórias 'match' e/ou 'similarity' ausentes.")

        X = df[["similarity"]]
        y = df["match"]

        logger.info(f"/models/train_model.py - Registros recebidos: {len(df)}")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        logger.info(f"/models/train_model.py - Divisão em treino ({len(X_train)}) e validação ({len(X_val)})")

        modelo = RandomForestClassifier(random_state=42, n_jobs=-1)

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, None],
            "min_samples_split": [2, 5],
        }

        logger.info("/models/train_model.py - Iniciando GridSearchCV com 3 folds...")
        grid_search = GridSearchCV(
            estimator=modelo,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            verbose=1,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        logger.info(f"/models/train_model.py - Melhores hiperparâmetros: {grid_search.best_params_}")

        y_pred = best_model.predict(X_val)
        y_proba = best_model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        roc = roc_auc_score(y_val, y_proba)

        logger.info(f"/models/train_model.py - Acurácia: {acc:.4f}")
        logger.info(f"/models/train_model.py - ROC AUC: {roc:.4f}")
        logger.info(f"/models/train_model.py - Classification Report:\n" + classification_report(y_val, y_pred))

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        logger.info(f"/models/train_model.py - Modelo salvo em: {MODEL_PATH}")

        fim = time.time()
        logger.info(f"/models/train_model.py - Treinamento finalizado em {fim - inicio:.2f} segundos")
        logger.info(f"/models/train_model.py - Log completo salvo em: /logs/")
        

        if return_model:
            return best_model

    except Exception as e:
        logger.critical(f"/models/train_model.py - Erro ao treinar o modelo: {e}", exc_info=True)
        raise
