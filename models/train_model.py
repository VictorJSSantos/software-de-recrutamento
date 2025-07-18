import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score
)

MODEL_PATH = "models/modelo_match.pkl"

def treinar_modelo(df: pd.DataFrame, return_model: bool = False):
    logging.info("Separando features e target...")

    # Drop seguro para evitar erros
    colunas_para_dropar = ["match", "id_candidato", "id_vaga"]
    colunas_presentes = [col for col in colunas_para_dropar if col in df.columns]

    if "match" not in df.columns:
        raise ValueError("Coluna 'match' (target) não encontrada no dataframe.")

    X = df.drop(columns=colunas_presentes)
    y = df["match"]

    # Split com estratificação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    logging.info("Treinando modelo Random Forest...")
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    # Avaliação
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    logging.info("Avaliação do modelo:")
    logging.info("\n" + classification_report(y_test, y_pred))
    logging.info(f"ROC-AUC: {roc_auc:.4f}")

    # Salvar modelo apenas se não for teste
    if not return_model:
        os.makedirs("models", exist_ok=True)
        joblib.dump(modelo, MODEL_PATH)
        logging.info(f"Modelo salvo em {MODEL_PATH}")

    if return_model:
        metrics = {
            "accuracy": acc,
            "roc_auc": roc_auc
        }
        return modelo, metrics
