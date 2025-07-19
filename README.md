# Decision AI Recruiter

Solução de Inteligência Artificial para Recrutamento e Seleção, desenvolvida para o **Datathon - Projeto Final da Pós em Engenharia de Machine Learning - Grupo 41**, para a empresa fictícia **Decision**.  
O foco é automatizar e otimizar o processo de *match* entre candidatos(as) e vagas, acelerando a contratação com mais precisão e menos viés.



## Objetivo
Desenvolver um modelo de machine learning capaz de prever a compatibilidade entre candidatos e vagas com base em embeddings e dados estruturados.

---

## Como rodar localmente

### 1. Clone o projeto
 
    git clone https://github.com/SEU_USUARIO/decision-ai-recruiter.git
    cd decision-ai-recruiter

### 2. Crie o ambiente virtual
 
    python -m venv venv
 # Para Linux/macOS: 
    source venv/bin/activate

 # Para Windows: 
    venv/Scripts/activate
    

### 3. Instale as dependências:
    pip install -r requirements.txt

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
 
    uvicorn app.main:app --reload

Acesse a documentação da API:
    
    Swagger: http://localhost:8000/docs
    ReDoc: http://localhost:8000/redoc


## Rodar com Docker
# Ambiente de Desenvolvimento
    docker build -f Dockerfile.dev -t decision-dev .
    docker run -p 8000:8000 decision-dev
 
# Ambiente de Produção

    docker build -t decision-api .
    docker run -p 8000:8000 decision-api


## Docker

# Build Manual
    docker build -t decision-api .
    docker run --rm -p 8000:8000 decision-api


# Ambientes (dev e prod)
    # Subir ambiente dev
    ./deploy.sh dev up

    # Subir ambiente prod
    ./deploy.sh prod up

    # Parar
    ./deploy.sh dev down

    # Build/restart
    ./deploy.sh prod restart


As variáveis de ambiente estão em:
    .env.dev

    .env.prod
    
### Testes
 pytest tests/

### Estrutura do Projeto

```bash
decision-ai-recruiter/
│
├── app/                  # API FastAPI
│   ├── main.py
│   ├── model.py
│   └── schema.py
│
├── pipeline/             # Pré-processamento
│   ├── preprocessing.py
│   ├── embedding_utils.py
│   └── feature_engineering.py
│
├── models/               # Treinamento e modelo salvo
│   └── train_model.py
│
├── data/                 # Dados brutos e processados
│
├── tests/                # Testes automatizados
│
├── Dockerfile
├── Dockerfile.dev
├── requirements.txt
└── README.md
```    


### Requisitos Atendidos
    _Pipeline de dados estruturada
    _Geração de embeddings
    _Dataset com variável alvo match
    _Treinamento e avaliação do modelo
    _API com FastAPI (endpoint /predict)
    _Testes automatizados (pytest)
    _Dockerização (dev e prod)


## Autores:

    Tatiana M. Haddad – – [@TatiHaddad](https://github.com/TatiHaddad)
    Victor Santos - [@VictorJSSantos](https://github.com/VictorJSSantos)
    Felipe Bizarria - [@felipebizarria](https://github.com/felipebizarria)
