# Dockerfile.dev

FROM python:3.11-slim

# Instalações básicas
RUN apt-get update && apt-get install -y gcc

# Diretório da aplicação
WORKDIR /app

# Copia dependências e instala
COPY requirements-prod.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia todo o projeto
COPY . .

# Executa a API com reload automático
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]