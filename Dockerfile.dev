# Etapa 1: Baixar modelos e preparar app
FROM python:3.12-slim as downloader

WORKDIR /stage

RUN apt-get update && apt-get install -y git git-lfs && \
    git clone --depth 1 https://github.com/VictorJSSantos/software-de-recrutamento.git && \
    mkdir -p models && cp software-de-recrutamento/models/*.pkl models/ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Etapa 2: Imagem final de produção
FROM python:3.12-slim

WORKDIR /app

# Copia os modelos da etapa anterior
COPY --from=downloader /stage/models/ ./models/

# Copia o app e dependências
COPY requirements.txt ./
COPY app ./app

# Instala apenas as dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expor a porta do Uvicorn
EXPOSE 8000

# Executar o app com Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]