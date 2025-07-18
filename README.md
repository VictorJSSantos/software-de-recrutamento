# Decision AI Recruiter

Solução com Inteligência Artificial para automatizar e melhorar o processo de recrutamento da empresa Decision.  
Desenvolvido como projeto final da pós-graduação em Engenharia de Machine Learning.

## Componentes
- Treinamento de modelo preditivo de match entre vaga e candidato
- API FastAPI para predição
- Docker para empacotamento
- Monitoramento e testes automatizados

## Como rodar

```bash
# Criar ambiente
python -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Rodar API
uvicorn app.main:app --reload



# Decision AI Recruiter

Projeto desenvolvido no Datathon da Pós-Tech FIAP com foco em Inteligência Artificial para Recrutamento e Seleção.

## 📌 Objetivo
Desenvolver um modelo de machine learning capaz de prever a compatibilidade entre candidatos e vagas com base em embeddings e dados estruturados.

---

## 🚀 Como rodar localmente

### 1. Clone o projeto
```bash
git clone https://github.com/SEU_USUARIO/decision-ai-recruiter.git
cd decision-ai-recruiter


2. Crie o ambiente virtual

python -m venv venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate    # Windows


3. Instale as dependências
bash
Copiar
Editar
pip install -r requirements.txt
4. Execute a API
bash
Copiar
Editar
uvicorn app.main:app --reload
🐳 Rodar com Docker
Ambiente de Desenvolvimento
bash
Copiar
Editar
docker build -f Dockerfile.dev -t decision-dev .
docker run -p 8000:8000 decision-dev
Ambiente de Produção
bash
Copiar
Editar
docker build -t decision-api .
docker run -p 8000:8000 decision-api
✅ Testes
bash
Copiar
Editar
pytest tests/
📂 Estrutura do Projeto
graphql
Copiar
Editar
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
📌 Requisitos Atendidos
 Pipeline de dados estruturada

 Geração de embeddings

 Dataset com variável alvo match

 Treinamento e avaliação do modelo

 API com FastAPI (endpoint /predict)

 Testes automatizados (pytest)

 Dockerização (dev e prod)

📤 Como atualizar a branch dev
Altere para a branch dev:

bash
Copiar
Editar
git checkout dev
Sincronize com a main se necessário:

bash
Copiar
Editar
git merge main
Adicione seus arquivos:

bash
Copiar
Editar
git add .
git commit -m "feat: adiciona Dockerfile e README"
git push origin dev
👩‍💻 Autoria
Projeto desenvolvido por Tatiana Haddad e [Seu Nome] no Datathon FIAP Pós-Tech - Engenharia de ML.