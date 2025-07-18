import pytest
import pandas as pd
from pipeline.utils import limpar_texto
from pipeline.preprocessing import preprocessar_dados
from pipeline.loaders import carregar_applicants

def test_limpar_texto():
    assert limpar_texto(" Olá, Mundo! ") == "ola mundo"
    assert limpar_texto("Téstê c/ acentuação!") == "teste c acentuacao"
    assert limpar_texto(None) == ""
    assert limpar_texto("") == ""

def test_preprocessar_dados_cria_features():
    df = pd.DataFrame({
        "respostas": ["uma resposta simples", None],
        "nivel": ["junior", "senior"],
        "descricao_vaga": ["Dev Python", "Analista Dados"]
    })
    df_proc = preprocessar_dados(df)
    assert "respostas_num_palavras" in df_proc.columns
    assert df_proc["respostas_num_palavras"].tolist() == [3, 0]
    assert "nivel_encoded" in df_proc.columns
    assert df_proc["descricao_vaga"].iloc[0] == "dev python"

def test_carregar_applicants_retorna_df():
    df = carregar_applicants()
    # Só testa se retornou DataFrame e tem colunas esperadas
    assert isinstance(df, pd.DataFrame)
    assert any(col.startswith("infos_basicas.") for col in df.columns)
