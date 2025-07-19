import pandas as pd

def test_dataset_final_integrity():
    df = pd.read_csv("data/processed/dataset_final.csv")
    assert not df.empty, "Dataset final está vazio"
    expected_columns = ["match"]
    for col in expected_columns:
        assert col in df.columns, f"Coluna obrigatória '{col}' ausente"
