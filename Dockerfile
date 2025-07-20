# FROM python:3.12-slim

# WORKDIR /app

# # Instala apenas os pacotes necessários para scikit-learn
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     python3-dev \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# COPY . .

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

### Abaixo a build de 1,66GB>
# FROM python:3.12-slim

# WORKDIR /app

# # Instalar git e git-lfs e baixar apenas os arquivos .pkl do HF
# RUN apt-get update && apt-get install -y git git-lfs && \
#     git lfs install && \
#     git clone https://huggingface.co/datasets/victormvll/software-de-recrutamento hf_tmp && \
#     mkdir -p models && cp hf_tmp/models/*.pkl models/ && \
#     rm -rf hf_tmp && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# # Copiar seu código do GitHub (sem os .pkl)
# COPY requirements.txt .
# COPY . .

# RUN pip install --upgrade pip && pip install -r requirements_v2.txt

# EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


#### Abaixo conseguimos 1,32GB:
# FROM python:3.12-slim AS builder

# WORKDIR /app

# # Instalar ferramentas para baixar os dados
# RUN apt-get update && apt-get install -y git git-lfs && \
#     git lfs install && \
#     git clone https://huggingface.co/datasets/victormvll/software-de-recrutamento && \
#     mkdir -p models && cp software-de-recrutamento/models/*.pkl models/

# # Copia apenas o necessário
# COPY requirements_v2.txt .

# # Instala as dependências em diretório isolado
# RUN pip install --upgrade pip && pip install --prefix=/install -r requirements_v2.txt


# # ---------------- Fase final ----------------

# FROM python:3.12-slim AS final

# WORKDIR /app

# # Copia pacotes já instalados do builder
# COPY --from=builder /install /usr/local

# # Copia apenas o necessário da app
# COPY . .

# # Copia modelos
# COPY --from=builder /app/models ./models

# EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Etapa 1: Baixar apenas os arquivos necessários
FROM python:3.12-slim as downloader

WORKDIR /stage

RUN apt-get update && apt-get install -y git git-lfs && \
    git lfs install && \
    git clone https://huggingface.co/datasets/victormvll/software-de-recrutamento && \
    mkdir -p models && cp software-de-recrutamento/models/*.pkl models/ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Etapa 2: Build final com o app
FROM python:3.12-slim

WORKDIR /app

# Copiar apenas os .pkl da etapa anterior
COPY --from=downloader /stage/models/ ./models/

# Copiar só os arquivos da aplicação (sem os .pkl nem o repositório clonado)
COPY requirements-prod.txt ./
COPY app ./app

# Instalar dependências
RUN pip install --upgrade pip && pip install -r requirements-prod.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
