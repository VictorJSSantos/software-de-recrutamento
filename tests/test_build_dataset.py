from pipeline.dataset_builder import build_dataset

if __name__ == "__main__":
    build_dataset(
        applicants_path="data/processed/applicants_processed.csv",
        vagas_path="data/processed/vagas_processed.csv",
        prospect_path="data/processed/prospects_processed.csv",
        output_path="data/processed/dataset_final.csv",
        negatives_ratio=1  # pode aumentar para mais exemplos negativos
    )
