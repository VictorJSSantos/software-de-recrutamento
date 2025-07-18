import os
import json
import pandas as pd
import re
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

RAW_PATH = "data/raw"  # Ajuste o caminho se precisar

def limpar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = unidecode(str(texto).lower())
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

def carregar_vagas_com_embeddings():
    caminho = os.path.join(RAW_PATH, "vagas.json")
    with open(caminho, "r", encoding="utf-8") as f:
        data = json.load(f)

    registros = []
    for codigo_vaga, vaga_info in data.items():
        registro = {}
        registro["codigo_vaga"] = codigo_vaga
        registro.update(vaga_info.get("informacoes_basicas", {}))
        registro.update(vaga_info.get("perfil_vaga", {}))
        registro.update(vaga_info.get("beneficios", {}))
        registros.append(registro)

    df = pd.DataFrame(registros)
    logging.info(f"vagas.json carregado com {len(df)} registros.")

    # Colunas que queremos unificar no texto
    colunas_para_unificar = [
        "titulo_vaga",
        "cliente",
        "solicitante_cliente",
        "empresa_divisao",
        "tipo_contratacao",
        "nivel profissional",
        "nivel_academico",
        "descricao_vaga",
        "tecnologias_desejadas",
        "principais_atividades",
        "competencia_tecnicas_e_comportamentais",
        "demais_observacoes",
    ]

    # Verifica se colunas existem no DF para evitar erro
    colunas_existentes = [c for c in colunas_para_unificar if c in df.columns]

    # Criar texto unificado concatenando as colunas existentes
    df['texto_unificado'] = df[colunas_existentes].fillna('').astype(str).agg(' '.join, axis=1)

    # Limpar o texto
    df['texto_unificado'] = df['texto_unificado'].apply(limpar_texto)

    # Gerar embeddings TF-IDF
    vectorizer = TfidfVectorizer(max_features=300)
    X = vectorizer.fit_transform(df['texto_unificado'])
    embeddings_df = pd.DataFrame(X.toarray(), columns=[f'embedding_{i}' for i in range(X.shape[1])])

    # Juntar embeddings no DF original
    df_final = pd.concat([df.reset_index(drop=True), embeddings_df.reset_index(drop=True)], axis=1)

    logging.info("Embeddings TF-IDF gerados para vagas.")
    return df_final

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df_vagas = carregar_vagas_com_embeddings()
    print(df_vagas.head())
