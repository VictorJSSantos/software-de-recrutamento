Objetivo do Projeto:
Desenvolver uma solução de IA para melhorar o processo de recrutamento e seleção da empresa Decision.

A Decision é especializada em serviços de bodyshop e recrutamento, com 
foco em conectar talentos qualificados às necessidades específicas dos clientes. 
A Decision atua principalmente no setor de TI, em que a agilidade e a precisão 
no “match” entre candidatos(as) e vagas são diferenciais essenciais. O objetivo 
da empresa é entregar profissionais que não apenas atendam aos requisitos 
técnicos, mas também se alinhem à cultura e aos valores das empresas 
contratantes.  

Foco para solução do problema da empresa:  Sistema de Recomendação e Classificação de Engajamento
_Otimizar o match entre os candidatos e as vagas
_Garantir o fit técnico e cultural
_Identificar engajamento/motivação


Análise Exploratória de Dados (EDA)
FOi feita análise nas 3 bases que estão no diretório:
software-de-recrutamento/data/applicants.json
software-de-recrutamento/data/vagas.json
software-de-recrutamento/data/prospects.json

Etapas:
1) EDA
    1.1) Importação e Carregamento
    1.2) Avaliação Estrutura e tipos
    1.3) Distribuição das variáveis:
        níveis academico x idiomas
        áreas de atuação x título profissionais
    1.4) Avaliação dados das vagas
    1.5) Distribuição das variáveis:
        contratado x não contratado

2) Pré Processamento e Feature Engineering
    2.1) Limpeza e tratamento dos dados ausentes
    2.2) Codificação das variáveis categóricas 
        _ OneHot
        _ Embeddings
    2.3) Transformação do texto com TF-IDF
    2.4) Feature Engineering:
        _ Última Experiência na área
        _ Similaridade job description
        _ Tempo Médio de resposta
        _ Participações em entrevista
        _ Linguagem Técnica



