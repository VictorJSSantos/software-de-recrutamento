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
