import os
import json
import pandas as pd
import logging

RAW_PATH = "data/raw"

def carregar_prospects():
    caminho = os.path.join(RAW_PATH, "prospects.json")
    with open(caminho, "r", encoding="utf-8") as f:
        data = json.load(f)

    registros = []
    for codigo_vaga, info_vaga in data.items():
        titulo = info_vaga.get("titulo", "")
        modalidade = info_vaga.get("modalidade", "")
        prospects = info_vaga.get("prospects", [])

        for prospect in prospects:
            prospect["codigo_vaga"] = codigo_vaga
            prospect["titulo_vaga"] = titulo
            prospect["modalidade_vaga"] = modalidade
            registros.append(prospect)

    df = pd.DataFrame(registros)
    logging.info(f"prospects.json carregado com {len(df)} registros.")
    return df

def carregar_applicants():
    caminho = os.path.join(RAW_PATH, "applicants.json")
    with open(caminho, "r", encoding="utf-8") as f:
        data = json.load(f)

    registros = []
    for codigo_candidato, candidato_info in data.items():
        registro = {}
        # Infos básicas
        for k, v in candidato_info.get("infos_basicas", {}).items():
            registro[f"infos_basicas.{k}"] = v

        # Informações pessoais
        for k, v in candidato_info.get("informacoes_pessoais", {}).items():
            registro[f"informacoes_pessoais.{k}"] = v

        # Informações profissionais
        for k, v in candidato_info.get("informacoes_profissionais", {}).items():
            registro[f"informacoes_profissionais.{k}"] = v

        # Formação e idiomas
        for k, v in candidato_info.get("formacao_e_idiomas", {}).items():
            registro[f"formacao_e_idiomas.{k}"] = v

        # CV PT e EN
        registro["cv_pt"] = candidato_info.get("cv_pt", "")
        registro["cv_en"] = candidato_info.get("cv_en", "")

        registro["codigo_candidato"] = codigo_candidato
        registros.append(registro)

    df = pd.DataFrame(registros)
    logging.info(f"applicants.json carregado com {len(df)} registros.")
    return df

def carregar_vagas():
    caminho = os.path.join(RAW_PATH, "vagas.json")
    with open(caminho, "r", encoding="utf-8") as f:
        data = json.load(f)

    registros = []
    for codigo_vaga, vaga_info in data.items():
        registro = {}
        registro["codigo_vaga"] = codigo_vaga
        
        for k, v in vaga_info.get("informacoes_basicas", {}).items():
            registro[f"infos_basicas.{k}"] = v
        
        for k, v in vaga_info.get("perfil_vaga", {}).items():
            registro[f"perfil_vaga.{k}"] = v
        
        for k, v in vaga_info.get("beneficios", {}).items():
            registro[f"beneficios.{k}"] = v
        
        registros.append(registro)

    df = pd.DataFrame(registros)
    logging.info(f"vagas.json carregado com {len(df)} registros.")
    return df
