import argparse
from pipeline.main import main

def rodar_pipeline():
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa pipeline de pré-processamento")
    parser.add_argument("--auto", action="store_true", help="Rodar automaticamente sem interação")
    args = parser.parse_args()

    try:
        rodar_pipeline()
        print("Pipeline executado com sucesso.")
    except Exception as e:
        print(f"Erro na execução do pipeline: {e}")
