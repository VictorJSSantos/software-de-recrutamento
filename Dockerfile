# FROM python:3.12-slim as downloader

# WORKDIR /stage

# RUN apt-get update && apt-get install -y git git-lfs && \
#     git lfs install && \
#     git clone https://huggingface.co/datasets/victormvll/software-de-recrutamento && \
#     mkdir -p models && cp software-de-recrutamento/models/*.pkl models/ && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# # Etapa 2: Build final com o app
# FROM python:3.12-slim

# WORKDIR /app

# # Copiar apenas os .pkl da etapa anterior
# COPY --from=downloader /stage/models/ ./models/

# # Copiar só os arquivos da aplicação (sem os .pkl nem o repositório clonado)
# COPY requirements-prod.txt ./
# COPY app ./app

# # Instalar dependências
# RUN pip install --upgrade pip && pip install -r requirements-prod.txt

# EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

##### Adicionando NGINX para poder usar proxy reverso e 
FROM python:3.12-slim as downloader

WORKDIR /stage

RUN apt-get update && apt-get install -y git git-lfs && \
    git lfs install && \
    git clone https://huggingface.co/datasets/victormvll/software-de-recrutamento && \
    mkdir -p models && cp software-de-recrutamento/models/*.pkl models/ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Etapa 2: Build final com o app e Nginx
FROM python:3.12-slim

WORKDIR /app

# Instala dependências do sistema e Nginx
RUN apt-get update && apt-get install -y nginx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar apenas os .pkl da etapa anterior
COPY --from=downloader /stage/models/ ./models/

# Copiar app e requirements
COPY requirements-prod.txt ./
COPY app ./app

# Instalar dependências do Python
RUN pip install --upgrade pip && pip install -r requirements-prod.txt

# Copiar configuração customizada do Nginx
COPY nginx.conf /etc/nginx/nginx.conf

# Expor portas do Nginx
EXPOSE 8000
EXPOSE 9090

# Rodar Nginx e Uvicorn simultaneamente
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & nginx -g 'daemon off;'"]
