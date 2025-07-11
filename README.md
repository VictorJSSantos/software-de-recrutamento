# Decision AI Recruiter

Solução de Inteligência Artificial para recrutamento e seleção, desenvolvida para o Datathon, Projeto da Pós de Engenharia de Machine Learning - Grupo 41, para a empresa **Decision**, com foco em automatizar e otimizar o processo de *match* entre candidatos(as) e vagas.


Esta aplicação implementa uma pipeline completa de Machine Learning para prever se um(a) candidato(a) tem perfil para ser contratado em uma vaga da Decision. Ela inclui:

-  Análise exploratória dos dados (EDA)
-  Pipeline de pré-processamento e engenharia de atributos
-  Modelo preditivo com Random Forest
-  API REST com FastAPI para inferência
-  Empacotamento com Docker
-  Testes unitários
-  Monitoramento com logging estruturado



##  Estrutura do Projeto

    
    software-de-recrutamento/
    ├── app/
    │   ├── main.py              # FastAPI com endpoint /predict
    │   ├── model.py             # Carregamento e inferência do modelo
    │   ├── schema.py            # Pydantic: validação da entrada e saída da API
    │
    ├── pipeline/
    │   ├── preprocessing.py     # Limpeza, encoding, TF-IDF
    │   ├── feature_engineering.py # Similaridade, experiência, etc.
    │
    ├── notebooks/
    │   ├── eda.ipynb            # Análise exploratória dos dados
    │
    ├── model/
    │   ├── model.joblib         # Modelo treinado
    │
    ├── tests/
    │   ├── test_preprocessing.py
    │   ├── test_feature_engineering.py
    │
    ├── data/
    │   ├── applicants.json
    │   ├── vagas.json
    │   ├── prospects.json
    │
    ├── train_model.py           # Script de treinamento
    ├── requirements.txt         # Dependências
    ├── Dockerfile               # Empacotamento Docker
    ├── .dockerignore
    └── README.md                # Este arquivo
    


## Requisitos:
    Python 3.12
    FastAPI
    scikit-learn
    pandas
    numpy
    joblib
    Docker

Instale as dependências com:
    pip install -r requirements.txt


## Como Treinar o Modelo:
    python train_model.py
O modelo será salvo em model/model.joblib


## Rodando os Testes:
    pytest tests/


## Rodando a API (local):
uvicorn app.main:app --reload

Acesse a documentação automática da API:
    Swagger UI: http://localhost:8000/docs
    Redoc: http://localhost:8000/redoc


## Rodando com Docker

Build da imagem:
    docker build -t decision-api .
    ou
    docker build --progress=plain -t decision-api .


Executar a API:
    docker run --rm -p 8000:8000 decision-api


## Endpoint de Previsão
POST /predict

    Exemplo da requisição:
        {
        "nivel_academico": "Superior Completo",
        "nivel_ingles": "Avançado",
        "nivel_espanhol": "Intermediário",
        "area_atuacao": "Desenvolvimento",
        "cv": "Profissional com 5 anos de experiência em Java, Spring Boot, e metodologias ágeis...",
        "descricao_vaga": "Buscamos dev com experiência em Java, APIs REST e conhecimento em cloud."
        }


    Resposta:
        {
        "match": true,
        "score": 0.87
        }


## Próximos passos e pendente:
_ Monitoramento e performance do modelo

_ registro dos logs com grafana


## Autores:
Tatiana M. Haddad – @TatiHaddad

Victor Santos - @VictorJSSantos

Felipe Bizarria - @felipebizarria
