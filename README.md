# Decision AI Recruiter

Solução de Inteligência Artificial para otimizar o processo de Recrutamento e Seleção, desenvolvida para o **Datathon - Projeto Final da Pós Graduação em Engenharia de Machine Learning (FIAP)**. O projeto foi elaborado pelo **Grupo 41** para a empresa **Decision**, especializada em alocação de talentos e body shop na área de tecnologia.


## Problema de Negócio

A empresa **Decision** enfrenta desafios crescentes para encontrar candidatos adequados de forma rápida, eficiente e com o menor viés possível. A triagem manual de currículos é **lenta, custosa** e muitas vezes **inconsistente**.


## Solução Proposta

Criamos uma solução baseada em Inteligência Artificial, para ser usada em uma plataforma, capaz de:
_ Analisar e comparar perfis de candidatos e descrição de vagas

_ Calcular a compatibilidade (match) entre os perfis com base em embeddings semânticos

_ Classificar automaticamente se um candidato é adequado para determinada vaga

_ Monitorar a performance do modelo e possíveis drifts ao longo do tempo


# Visão Geral da Arquitetura:
    A[Dados Brutos JSON] --> B[Pré-processamento]
                                B --> C[Geração de Embeddings com Sentence Transformers]
                                        C --> D[Dataset com Variável Target: Match]
                                                D --> E[Treinamento do Modelo]
                                                        E --> F[API FastAPI (endpoint /predict)]
                                                                F --> G[Monitoramento com Streamlit]



# Funcionalidades
        _ Pipeline de dados estruturada e modular
        _ Pré-processamento e vetorização de dados textuais
        _ Geração de embeddings semânticos com sentence-transformers
        _ Construção de um dataset com variável-alvo de match
        _ Modelo preditivo de classificação binárica (match / no match)
        _ API com FastAPI para consumo em tempo real
        _ Monitoramento contínuo dos logs com Streamlit
        _ Dockerização dos ambientes de desenvolvimento e produção


# Resultado Esperado
        _ Redução do tempo de triagem de currículos
        _ Aumento da assertividade na seleção
        _ Redução de vieses manuais
        _ Melhoria na experiência de candidatos e eficiência dos recrutadores


---

## Como rodar: 
Este projeto pode ser executado de três formas principais:
    1) Ambiente de Desenvolvimento com Docker
    2) Ambiente de Produção com Docker / Render
    3) Localmente sem Docker



### 1. Clone o projeto
  ```bash
    git clone https://github.com/VictorJSSantos/software-de-recrutamento.git
    cd software-de-recrutamento
  ```
  

### 2. Crie o ambiente virtual
   ```bash
    python -m venv venv
 # Para Linux/macOS: 
    source venv/bin/activate

 # Para Windows: 
    venv/Scripts/activate

  ```


### 3. Instale as dependências:
     ```bash
    pip install -r requirements.txt
    ```
    ```bash
     ```

# Requisitos:
    Python 3.12
    FastAPI
    scikit-learn
    pandas
    numpy
    joblib
    Docker + Docker Compose
    Pip




###4. Execute a API
 ```bash
    uvicorn app.main:app --reload
 ```

Acesse a documentação da API:
  ```bash 
    Swagger: http://localhost:8000/docs
    ReDoc: http://localhost:8000/redoc
 ```


### Previsão de Compatibilidade
    ### Endpoint: /predict
    Método: POST

        ### Request(JSON)
 ```bash
        {
        "descricao_candidato": "Desenvolvedor com experiência em Python, Django, APIs REST e SQL.",
        "descricao_vaga": "Buscamos engenheiro de software com domínio em Python, APIs RESTful e banco de dados relacional."
        }

 ```


        ### Response(JSON)
 ```bash
            {
            "similaridade": 0.8347,
            "match": true
            }

 ```


## Lógica de Similaridade e Match
_ Utiliza sentence-transformers para gerar embeddings vetorias dos textos
_ Similaridade é calculada com cosine_similarity entre os vetores da vaga e do candidato
_ O modelo classificador aprende a prever match = 1 ou 0 com base na similaridade e características adicionais
_ OS arquivos .pkl salvos:
    _ modelo_match.pkl: modelo treinado
    _ vectorizer.pkl: responsável por embeddings


## Métricas de Performance
Avaliação do modelo de classificação:
Métrica	Valor
Acurácia	0.92
Precisão	0.89
Recall	0.90
F1-Score	0.895
ROC AUC	0.94


## Logs de Predição
A cada requisição feita à API, um log é salvo em ./logs/, com:
timestamp, descricao_candidato, descricao_vaga, similaridade, match, tempo_execucao


## Dashboard de Monitoramento
Executado com Streamlit:
   ```bash
    streamlit run monitoramento/monitoramento_drift.py

 ```
Dashboard de monitoramento com Visualizações:
 Similaridade média por dia
 %Percentual de match por dia
 Alertas visuais se houver drift (variação significativa)
 Filtros por data ou intervalo (via st.date_input e st.slider)



## Rodar com Docker
# Ambiente de Desenvolvimento
   ```bash
    docker build -f Dockerfile.dev -t decision-dev .
    docker run -p 8000:8000 decision-dev
 ```

# Ambiente de Produção
```bash
    docker build -t decision-api .
    docker run -p 8000:8000 decision-api
```

## Docker

# Build Manual
   ```bash
    docker build -t decision-api .
    docker run --rm -p 8000:8000 decision-api
```


# Ambientes (dev e prod) - Usando Docker Compose
 ```bash
    # Subir ambiente dev
    ./deploy.sh dev up

    # Subir ambiente prod
    ./deploy.sh prod up

    # Parar
    ./deploy.sh dev down

    # Build/restart
    ./deploy.sh prod restart
```

As variáveis de ambiente estão em:
```bash
    .env.dev

    .env.prod
```    



### Testes Automatizados
```bash
$env:PYTHONPATH="."  # apenas para a sesão de teste
pytest tests/

 ```




### Estrutura do Projeto

```bash
decision-ai-recruiter/
│
├── ci.yml
│
├── app/                  # API FastAPI
│   ├── main.py
│   ├── model.py
│   └── schema.py
│
├── data/             # Dados Brutos e Processados
├── processed/             # Dados Processados
│   ├── applicants_processed.csv
│   ├── dataset_final.csv
│   ├── prospects_processed.csv
│   └── vagas_processed.csv
├── raw/             # Dados Brutos
│   ├── applicants.json
│   ├── prospects.json
│   └── vagas.json
│
├── logs/                 # Logs de predição para monitoramento
│   ├── 01_carregamento_20250719_221020.log
│   ├── 02_preprocessamento_20250719_221022.log
│   ├── 03_geracao_dataset_20250719_221027.log
│   ├── 04_treinamento_modelo_20250719_221029.log
│   ├── 05_api_20250719_221101.log
│   ├── 06_predictions_log_20250717_221022.log
│   ├── 06_predictions_log_20250718_221022.log
│   ├── 06_predictions_log_20250719_221022.log
│   ├── 06_predictions_log_20250720_221022.log
│   └── teste_treinamento_modelo_20250719_221029.log
│
├── models/               # Treinamento de modelo e modelos salvos
│   ├── modelo_match.pkl
│   ├── train_model.py
│   └── vectorizer.pkl
│
├── monitoramento/        # Coleta Monitoramento Drift e Dashboard Streamlit
│   └── logs/
│       └── 06_predictions_log.csv
│   ├── consolidar_logs.py
│   └── monitoramento_drift.py
│
├── notebook/             # Scripts para validação e EDA
│   └── exploratory_analysis.ipynb
│   └── EDA.ipynb
│
├── pipeline/             # Pipeline de dados
│   ├── dataset_builder.py
│   ├── load_vaga.py
│   ├── loaders.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── save.py
│   └── utils.py
│
├── scripts/               # Treinamento
│   └── train.py
│
├── tests/                # Testes unitários e de integração
│   └── test_api_integration.py
│   └── test_dataset_integrity.py
│   └── test_model_serialization.py
│   └── test_pipeline.py
│   └── test_train_model.py
│
├── run_pipeline.py     # Execução automatizada de todo o pipeline
├── Dockerfile          # Imagem de produção
├── Dockerfile.dev      # Imagem para desenvolvimento
├── docker-compose.dev.yml          
├── docker-compose.prod.yml
├── requirements.txt
├── .env.dev
├── .env.prod
├── .dockerignore
├── .gitignore
└── README.md
```    


### Requisitos Atendidos
    _ Pipeline completa com ingestão e pré-processamento de dados: limpeza
    _ Geração de embeddings com sentence-trnasformers e engenharia de features
    _ Dataset final com variável alvo *match*
    _ Modelo preditivo treinado e salvo (RandomForestClassifier)
    _ API funcional com FastAPI (endpoint /predict) e documentação Swagger
    _ Testes automatizados unitários e de integração
    _ Containerização com Docker
    _ Deploy na Produtivo Nuvem
    _ Monitoramento contínuo com logs e Streamlit e Grafana + Prometheus




## Autores:

   Tatiana M. Haddad – – [@TatiHaddad](https://github.com/TatiHaddad)
   Victor Santos - [@VictorJSSantos](https://github.com/VictorJSSantos)
   Felipe Bizarria - [@felipebizarria](https://github.com/felipebizarria)
