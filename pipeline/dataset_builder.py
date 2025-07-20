import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_embeddings(df, prefix="embedding_"):
    return df[[col for col in df.columns if col.startswith(prefix)]].values


def build_dataset(
    applicants_path, vagas_path, prospect_path, output_path, negatives_ratio=1
):
    logging.info("Carregando dados...")
    applicants_df = pd.read_csv(applicants_path)
    vagas_df = pd.read_csv(vagas_path)
    prospect_df = pd.read_csv(prospect_path)

    # Removendo duplicados
    applicants_df = applicants_df.drop_duplicates(subset=["codigo_candidato"])
    vagas_df = vagas_df.drop_duplicates(subset=["codigo_vaga"])

    # Embeddings
    applicant_embeddings = load_embeddings(applicants_df)
    vaga_embeddings = load_embeddings(vagas_df)

    # Indexação por código
    applicant_dict = applicants_df.set_index("codigo_candidato").to_dict(orient="index")
    vaga_dict = vagas_df.set_index("codigo_vaga").to_dict(orient="index")

    logging.info("Iniciando geracao do dataset com matches reais...")
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

    logging.info(f"Gerando negativos com razao {negatives_ratio}...")
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
    df_final.to_csv(output_path, index=False)
    logging.info(
        f"""Dataset final salvo com {df_final.shape[0]} registros em {output_path}\n
                    {df_final.columns}"""
    )
