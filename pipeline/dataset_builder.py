import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def load_embeddings(df, prefix="embedding_"):
    return df[[col for col in df.columns if col.startswith(prefix)]].values

def build_dataset(applicants_df, vagas_df, prospect_df, output_path=None, logger=None, negatives_ratio=1):
    logger = logger or logging.getLogger(__name__)    
    logger.info("/pipeline/dataset_builder.py - Iniciando construção do dataset...")

    # Removendo duplicados
    logger.info("/pipeline/dataset_builder.py - Remove duplicados applicants...")
    applicants_df = applicants_df.drop_duplicates(subset=['codigo_candidato'])
    logger.info("/pipeline/dataset_builder.py - Remove duplicados vagas...")
    vagas_df = vagas_df.drop_duplicates(subset=['codigo_vaga'])


    # Embeddings
    logger.info("/pipeline/dataset_builder.py - Embeddings applicants...")
    applicant_embeddings = load_embeddings(applicants_df)
    logger.info("/pipeline/dataset_builder.py - Embeddings vagas...")
    vaga_embeddings = load_embeddings(vagas_df) 
    
    # Indexação por código para acesso rápido
    logger.info("/pipeline/dataset_builder.py - Indexando applicants...")
    applicant_dict = applicants_df.set_index('codigo_candidato').to_dict(orient='index')
    logger.info("/pipeline/dataset_builder.py - Indexando vagas...")
    vaga_dict = vagas_df.set_index('codigo_vaga').to_dict(orient='index')

    dataset = []
    for _, row in tqdm(prospect_df.iterrows(), total=len(prospect_df)):
        cod_app = row["codigo"]
        cod_vaga = row["codigo_vaga"]

        if cod_app not in applicant_dict or cod_vaga not in vaga_dict:
            continue

        emb_app = np.array(
            [applicant_dict[cod_app][f"embedding_{i}"] for i in range(300)]
        ).reshape(1, -1)
        emb_vaga = np.array(
            [vaga_dict[cod_vaga][f"embedding_{i}"] for i in range(300)]
        ).reshape(1, -1)

        similarity = cosine_similarity(emb_app, emb_vaga)[0][0]
        dataset.append(
            {
                "codigo_candidato": cod_app,
                "codigo_vaga": cod_vaga,
                "similarity": similarity,
                "match": 1,
            }
        )

    logger.info(f"/pipeline/dataset_builder.py - Gerando negativos com razao {negatives_ratio}...")

    all_applicants = list(applicant_dict.keys())
    all_vagas = list(vaga_dict.keys())

    num_negativos = len(dataset) * negatives_ratio
    attempts = 0
    while (
        len(dataset) < len(prospect_df) + num_negativos
        and attempts < 10 * num_negativos
    ):
        app = np.random.choice(all_applicants)
        vaga = np.random.choice(all_vagas)

        if not (
            (prospect_df["codigo"] == app) & (prospect_df["codigo_vaga"] == vaga)
        ).any():
            emb_app = np.array(
                [applicant_dict[app][f"embedding_{i}"] for i in range(300)]
            ).reshape(1, -1)
            emb_vaga = np.array(
                [vaga_dict[vaga][f"embedding_{i}"] for i in range(300)]
            ).reshape(1, -1)
            similarity = cosine_similarity(emb_app, emb_vaga)[0][0]
            dataset.append(
                {
                    "codigo_candidato": app,
                    "codigo_vaga": vaga,
                    "similarity": similarity,
                    "match": 0,
                }
            )
        attempts += 1

    df_final = pd.DataFrame(dataset)

    if output_path:
        df_final.to_csv(output_path, index=False)
        logger.info(f"/pipeline/dataset_builder.py - Dataset final salvo com {df_final.shape[0]} registros em {output_path}")

    return df_final