#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
import re
import time
from azure.storage.blob import BlobServiceClient
#import fitz 
#from docx import Document
import pytesseract
from PIL import Image
import io
from azure.cosmos import CosmosClient, exceptions
import argparse
import pdfplumber
import io
import os
import logging
from pydantic import BaseModel
from langgraph.graph import StateGraph
import random
from typing import Literal
from langgraph.graph import START, END
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import spacy
from sentence_transformers import SentenceTransformer
#from annoy import AnnoyIndex
import faiss
from typing import Optional, Annotated, Dict, Any
from typing_extensions import Annotated
from pathlib import Path
import glob
import math


#============>>>>>>>>>=======Data=====<<<<<<<<<:


nltk.download('stopwords')
stemmer = SnowballStemmer('spanish')
stop_words = set(stopwords.words('spanish'))
palabras_irrelevantes = {'cordoba', 'distrital', 'solicitud', 'petición', 'ciudad', 'fiscalizacion'}  #Palabras sin significado clave
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("spanish"))
#nlp = spacy.load("es_core_news_sm")  # Modelo de spaCy para español
nlp = spacy.load("es_core_news_sm")  # Modelo de spaCy para español

nltk.download('wordnet')
#nltk.download('stopwords')
nltk.download('stopwords')
stemmer = SnowballStemmer('spanish')

#Inicializar el modelo de embeddings
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#Entidades de referencia y convertimos a tipo float32 para FAISS
entidades = ["Fiscalización caso 1", "Fiscalización caso 2", "Fiscalización caso 3", "Fiscalización caso 4"]
categoria_embeddings = embedding_model.encode(entidades)
categoria_embeddings = np.array(categoria_embeddings).astype('float32')

d = categoria_embeddings.shape[1]  #Dimensión de los embeddings
faiss_index = faiss.IndexFlatL2(d)  #Índice L2 (distancia euclidiana)
faiss_index.add(categoria_embeddings)  #Agregar los embeddings al índice

faiss.write_index(faiss_index, "faiss_index.bin")

#Credenciales OpenAI:
openai.api_type = "azure"
openai.api_base = "https://ai-danielvallejosai0530531063088720.openai.azure.com/"
openai.api_version = "2024-12-01-preview"
openai.api_key = "EARAfege87W1E3y9jdxiIxopGCUJkXclbIe3AgF55j4K8u6SW77eJQQJ99BDACHYHv6XJ3w3AAAAACOGYMCD"

embedding_deployment_name = "text-embedding-3-large"
embedding_api_version = "2023-05-15"
chat_deployment_name = "gpt-4.1"

client = openai.AzureOpenAI(  # Usa AzureOpenAI en lugar de Client
api_key="EARAfege87W1E3y9jdxiIxopGCUJkXclbIe3AgF55j4K8u6SW77eJQQJ99BDACHYHv6XJ3w3AAAAACOGYMCD",
azure_endpoint="https://ai-danielvallejosai0530531063088720.openai.azure.com/",
api_version="2024-12-01-preview")


#============>>>>>>>>>=======queryText=====<<<<<<<<<:

#Text:
def leer_archivos_txt_a_dataframe(ruta_base, incluir_subcarpetas=True, guardar_csv=True):    

    ruta_base = Path(ruta_base)
    datos = []

    if incluir_subcarpetas:
        patron = "**/*.txt"  
    else:
        patron = "*.txt"  

    archivos_txt = list(ruta_base.glob(patron))

    if not archivos_txt:
        print("No se encontraron archivos TXT en la ruta especificada")
        return pd.DataFrame(columns=['Case', 'Description'])

    for i, archivo in enumerate(archivos_txt, 1):
        try:
            print(f"Procesando {i}/{len(archivos_txt)}: {archivo.name}")

            with open(archivo, 'r', encoding='utf-8') as f:
                contenido = f.read()

            nombre_caso = archivo.stem

            datos.append({
                'Case': nombre_caso,
                'Description': contenido
            })

            print(f"Leído exitosamente ({len(contenido)} caracteres)")

        except Exception as e:
            print(f"Error leyendo {archivo.name}: {str(e)}")

            datos.append({
                'Case': archivo.stem,
                'Description': f"ERROR AL LEER ARCHIVO: {str(e)}"
            })

    df = pd.DataFrame(datos)

    if not df.empty:
        df_clean = df[~df['Description'].str.startswith('ERROR', na=False)]
        if not df_clean.empty:
            print(f"Promedio de caracteres por descripción: {df_clean['Description'].str.len().mean():.0f}")
            print(f"Descripción más larga: {df_clean['Description'].str.len().max()} caracteres")
            print(f"Descripción más corta: {df_clean['Description'].str.len().min()} caracteres")

    return df

#============>>>>>>>>>=======Consolidade=====<<<<<<<<<:

ruta_txt_aporte = "/Users/rayespinoza/AI_Projects/cordoba_classificator/data/casos/extract/Caso_1/aporte/"
ruta_txt_corrida = "/Users/rayespinoza/AI_Projects/cordoba_classificator/data/casos/extract/Caso_1/corrida/"
df_aporte = leer_archivos_txt_a_dataframe(
    ruta_base=ruta_txt_aporte,
    incluir_subcarpetas=False,
    guardar_csv=False
)
df_corrida = leer_archivos_txt_a_dataframe(
    ruta_base=ruta_txt_corrida,
    incluir_subcarpetas=False, 
    guardar_csv=False
)
df_caso1 = pd.concat([df_aporte, df_corrida], ignore_index=True)

#Combine the DataFrames:
text_columns = [col for col in df_caso1.columns if col.startswith('text')]

if text_columns:
    query_text = ' '.join(df_caso1[text_columns].fillna('').astype(str).values.flatten())
else:
    content_columns = [col for col in df_caso1.columns if col in ['contenido', 'texto', 'content', 'text']]

    if content_columns:
        query_text = ' '.join(df_caso1[content_columns].fillna('').astype(str).values.flatten())
    else:
        string_columns = df_caso1.select_dtypes(include=['object', 'string']).columns.tolist()
        query_text = ' '.join(df_caso1[string_columns].fillna('').astype(str).values.flatten())
#Limpiar el texto:
query_text = ' '.join(query_text.split())

print(f"Query text generado con {len(query_text)} caracteres")
print(f"Primeros 200 caracteres: {query_text[:200]}...")


#============>>>>>>>>>=======Start LangGraph=====<<<<<<<<<:

def get_openai_client():
    openai.api_type = "azure"
    openai.api_base = "https://ai-danielvallejosai0530531063088720.openai.azure.com/"
    openai.api_version = "2024-12-01-preview"
    openai.api_key = "EARAfege87W1E3y9jdxiIxopGCUJkXclbIe3AgF55j4K8u6SW77eJQQJ99BDACHYHv6XJ3w3AAAAACOGYMCD"

    client = openai.AzureOpenAI(
        api_key="EARAfege87W1E3y9jdxiIxopGCUJkXclbIe3AgF55j4K8u6SW77eJQQJ99BDACHYHv6XJ3w3AAAAACOGYMCD",
        azure_endpoint="https://ai-danielvallejosai0530531063088720.openai.azure.com/",
        api_version="2024-12-01-preview"
    )

    return client, {
        "embedding_deployment_name": "text-embedding-3-large",
        "embedding_api_version": "2023-05-15",
        "chat_deployment_name": "gpt-4.1"
    }


class GraphState(BaseModel):
    graph_nropeticion: Optional[str] = ""
    graph_contexto: str
    graph_state: Optional[str] = ""
    graph_clasificacionIni: Optional[str] = ""
    graph_normalizacion: Optional[str] = ""
    graph_similitud_coseno: Optional[str] = ""
    graph_summarize_pqrs: Optional[str] = ""

    graph_case1_pqrs: Optional[str] = ""
    graph_case2_pqrs: Optional[str] = ""
    graph_case3_pqrs: Optional[str] = ""
    graph_case4_pqrs: Optional[str] = ""
    graph_case5_pqrs: Optional[str] = ""
    graph_case6_pqrs: Optional[str] = ""
    graph_case7_pqrs: Optional[str] = ""
    graph_case8_pqrs: Optional[str] = ""

    graph_clasificacionPre: Optional[str] = ""
    graph_clasificacionFinal: Optional[str] = ""

    graph_revision_humana: Optional[str] = ""

def greeting_node(state):
    #Iniciamos el modelo de GraphState:
    return {}

def peticion_node(state):
    #Solo modificamos graph_state:
    return {"graph_state": "Petición recibida"}


def limpieza_normalizacion_node2(state):
    query_text=state.graph_contexto

    texto = query_text.lower()
    texto = re.sub(r'\W', ' ', texto)
    texto = re.sub(r'\d+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    #===>Tokenización y lematización
    palabras = texto.split()
    #palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra not in stop_words]
    palabras = [stemmer.stem(palabra) for palabra in palabras if palabra not in stop_words and palabra not in palabras_irrelevantes]  # Stemming
    texto_procesado = ' '.join(palabras)
    #===>Extracción de entidades nombradas (NER)
    doc = nlp(query_text)
    entidades = [ent.text for ent in doc.ents]
    #state.graph_normalizacion=texto_procesado
    return {"graph_normalizacion": texto_procesado}

def semantica_entidades_node3(state):

    #New:
    texto_procesado=state.graph_normalizacion
    #texto_procesado = datos["texto_procesado"]
    #Generar embedding del texto
    embedding = embedding_model.encode([texto_procesado])

    #Buscar en FAISS la categoría más cercana
    D, I = faiss_index.search(np.array(embedding).astype('float32'), k=3)  #Buscar top 3 entidades

    #Obtener las entidades
    entidades = ["Fiscalización caso 1", "Fiscalización caso 2", "Fiscalización caso 3", "Fiscalización caso 4"]
    entidades_similares = [entidades[i] for i in I[0]]

    #state.graph_similitud_coseno=entidades_similares
    return {"graph_similitud_coseno": entidades_similares}

def summarize_entidades_node4(state):

    #To sum up:
    query_text=state.graph_normalizacion

    client, config = get_openai_client()

    with open('/Users/rayespinoza/Visual Code/LangchainRAG/PROMPT_SUMMARIZE_EN.txt', 'r', encoding='utf-8') as file:
        PROMPT_TEMPLATE_G = file.read()

    prompt = PROMPT_TEMPLATE_G.format(
            question=query_text
        )

    response = client.chat.completions.create(
                model=chat_deployment_name,
                messages=[
                    {"role": "system", "content": "Eres un experto en resumiendo peticiones."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1
            )

    resultado = response.choices[0].message.content.strip()

    return {"graph_summarize_pqrs": resultado}



def clasificacion_caso1(state):

    query_text=state.graph_normalizacion
    resumen_pqrs=state.graph_summarize_pqrs
    entidades_coseno=state.graph_similitud_coseno

    client, config = get_openai_client()

    resultados = []
    with open('/Users/rayespinoza/Visual Code/LangchainRAG/PROMPT_GOBIERNO_PQRS_EN.txt', 'r', encoding='utf-8') as file:
        PROMPT_TEMPLATE_G = file.read()

    prompts_entidades = [
    ("200", PROMPT_TEMPLATE_G) #Secretaría de Gobierno
    ] #Secretaría de Integración Social

    prompt = PROMPT_TEMPLATE_G.format(
            question=query_text,
            resumen=resumen_pqrs,
            entidad_similitud=entidades_coseno)

    response = client.chat.completions.create(
                model=chat_deployment_name,
                messages=[
                    {"role": "system", "content": "Eres un experto en clasificación de entidades."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1,
                #max_tokens=100,
                logprobs=True,
                top_logprobs=5
            )

    #Obtener respuesta:
    resultado = response.choices[0].message.content.strip()

    resultados.append({
                "requirement_code": numero_query,
                "codigoEntidad": prompts_entidades[0][0],
                "resultado": int(resultado)
            })

    return {"graph_entidad_gobierno_pqrs": resultados}

def clasificacion_caso2(state):

    query_text=state.graph_normalizacion
    resumen_pqrs=state.graph_summarize_pqrs
    entidades_coseno=state.graph_similitud_coseno

    client, config = get_openai_client()


    resultados = []
    with open('/Users/rayespinoza/Visual Code/LangchainRAG/PROMPT_MOVILIDAD_PQRS_EN.txt', 'r', encoding='utf-8') as file:
        PROMPT_TEMPLATE_M = file.read()

    prompts_entidades = [
    ("1100", PROMPT_TEMPLATE_M) #Secretaría de Movilidad
    ]

    prompt = PROMPT_TEMPLATE_M.format(
            question=query_text,
            resumen=resumen_pqrs,
            entidad_similitud=entidades_coseno)

    response = client.chat.completions.create(
                model=chat_deployment_name,
                messages=[
                    {"role": "system", "content": "Eres un experto en clasificación de entidades."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1,
                #max_tokens=100,
                logprobs=True,
                top_logprobs=5
            )

    #Obtener respuesta:
    resultado = response.choices[0].message.content.strip()

    resultados.append({
                "requirement_code": numero_query,
                "codigoEntidad": prompts_entidades[0][0],
                "resultado": int(resultado)
            })

    return {"graph_entidad_movilidad_pqrs": resultados}

def clasificacion_caso3(state):

    query_text=state.graph_normalizacion
    resumen_pqrs=state.graph_summarize_pqrs
    entidades_coseno=state.graph_similitud_coseno

    client, config = get_openai_client()

    resultados = []
    with open('/Users/rayespinoza/Visual Code/LangchainRAG/PROMPT_IS_PQRS_EN.txt', 'r', encoding='utf-8') as file: 
        PROMPT_TEMPLATE_IS = file.read()

    prompts_entidades = [
    ("800", PROMPT_TEMPLATE_IS)] #Secretaría de Integración Social

    prompt = PROMPT_TEMPLATE_IS.format(
            question=query_text,
            resumen=resumen_pqrs,
            entidad_similitud=entidades_coseno)

    response = client.chat.completions.create(
                model=chat_deployment_name,
                messages=[
                    {"role": "system", "content": "Eres un experto en clasificación de entidades."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1,
                #max_tokens=100,
                logprobs=True,
                top_logprobs=5
            )

    #Obtener respuesta:
    resultado = response.choices[0].message.content.strip()

    resultados.append({
                "requirement_code": numero_query,
                "codigoEntidad": prompts_entidades[0][0],
                "resultado": int(resultado)
            })

    return {"graph_entidad_is_pqrs": resultados}

def clasificacion_caso4(state):

    query_text=state.graph_normalizacion
    resumen_pqrs=state.graph_summarize_pqrs
    entidades_coseno=state.graph_similitud_coseno

    client, config = get_openai_client()

    resultados = []
    with open('/Users/rayespinoza/Visual Code/LangchainRAG/PROMPT_SALUD_PQRS_EN.txt', 'r', encoding='utf-8') as file: 
        PROMPT_TEMPLATE_S = file.read()

    prompts_entidades = [
    ("700", PROMPT_TEMPLATE_S), #Secretaría de Salud
    ]

    prompt = PROMPT_TEMPLATE_S.format(
            question=query_text,
            resumen=resumen_pqrs,
            entidad_similitud=entidades_coseno)

    response = client.chat.completions.create(
                model=chat_deployment_name,
                messages=[
                    {"role": "system", "content": "Eres un experto en clasificación de entidades."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1,
                #max_tokens=100,
                logprobs=True,
                top_logprobs=5
            )

    #Obtener respuesta:
    resultado = response.choices[0].message.content.strip()

    resultados.append({
                "requirement_code": numero_query,
                "codigoEntidad": prompts_entidades[0][0],
                "resultado": int(resultado)
            })

    return {"graph_entidad_salud_pqrs": resultados}


def clasificacion_caso5(state):

    query_text=state.graph_normalizacion
    resumen_pqrs=state.graph_summarize_pqrs
    entidades_coseno=state.graph_similitud_coseno

    client, config = get_openai_client()

    resultados = []
    with open('/Users/rayespinoza/Visual Code/LangchainRAG/sector/GestionPublica/PROMPT_SECTOR_GESTIONPUBLICA_PQRS.txt', 'r', encoding='utf-8') as file: 
        PROMPT_TEMPLATE_GP = file.read()

    prompts_entidades = [
    ("001", PROMPT_TEMPLATE_GP), #Secretaría de Salud
    ]

    prompt = PROMPT_TEMPLATE_GP.format(
            question=query_text,
            resumen=resumen_pqrs,
            entidad_similitud=entidades_coseno)

    response = client.chat.completions.create(
                model=chat_deployment_name,
                messages=[
                    {"role": "system", "content": "Eres un experto en clasificación de entidades."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1,
                #max_tokens=100,
                logprobs=True,
                top_logprobs=5
            )

    #Obtener respuesta:
    resultado = response.choices[0].message.content.strip()

    resultados.append({
                "requirement_code": numero_query,
                "codigoEntidad": prompts_entidades[0][0],
                "resultado": int(resultado)
            })

    return {"graph_clasificacionsectorGestionPublica": resultados}


def clasificacion_caso6(state):

    query_text=state.graph_normalizacion
    resumen_pqrs=state.graph_summarize_pqrs
    entidades_coseno=state.graph_similitud_coseno

    client, config = get_openai_client()

    resultados = []
    with open('/Users/rayespinoza/Visual Code/LangchainRAG/sector/Gobierno/PROMPT_SECTOR_GOBIERNO_PQRS.txt', 'r', encoding='utf-8') as file: 
        PROMPT_TEMPLATE_GP = file.read()

    prompts_entidades = [
    ("002", PROMPT_TEMPLATE_GP), #Secretaría de Salud
    ]

    prompt = PROMPT_TEMPLATE_GP.format(
            question=query_text,
            resumen=resumen_pqrs,
            entidad_similitud=entidades_coseno)

    response = client.chat.completions.create(
                model=chat_deployment_name,
                messages=[
                    {"role": "system", "content": "Eres un experto en clasificación de entidades."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1,
                #max_tokens=100,
                logprobs=True,
                top_logprobs=5
            )

    #Obtener respuesta:
    resultado = response.choices[0].message.content.strip()

    resultados.append({
                "requirement_code": numero_query,
                "codigoEntidad": prompts_entidades[0][0],
                "resultado": int(resultado)
            })

    return {"graph_clasificacionsectorGobierno": resultados}

def clasificacion_caso7(state):

    query_text=state.graph_normalizacion
    resumen_pqrs=state.graph_summarize_pqrs
    entidades_coseno=state.graph_similitud_coseno

    client, config = get_openai_client()

    client = openai.AzureOpenAI(  # Usa AzureOpenAI en lugar de Client
    api_key="b427f4f746d24d5fa2853dc8f16979cd",
    azure_endpoint="https://oai-inboxia-poc.openai.azure.com/",
    api_version="2024-08-01-preview")

    resultados = []
    with open('/Users/rayespinoza/Visual Code/LangchainRAG/sector/Hacienda/PROMPT_SECTOR_HACIENDA_PQRS.txt', 'r', encoding='utf-8') as file: 
        PROMPT_TEMPLATE_GP = file.read()

    prompts_entidades = [
    ("003", PROMPT_TEMPLATE_GP),
    ]

    prompt = PROMPT_TEMPLATE_GP.format(
            question=query_text,
            resumen=resumen_pqrs,
            entidad_similitud=entidades_coseno)

    response = client.chat.completions.create(
                model=chat_deployment_name,
                messages=[
                    {"role": "system", "content": "Eres un experto en clasificación de entidades."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1,
                #max_tokens=100,
                logprobs=True,
                top_logprobs=5
            )

    #Obtener respuesta:
    resultado = response.choices[0].message.content.strip()

    resultados.append({
                "requirement_code": numero_query,
                "codigoEntidad": prompts_entidades[0][0],
                "resultado": int(resultado)
            })

    return {"graph_clasificacionsectorHacienda": resultados}


def clasificacion_caso8(state):

    query_text=state.graph_normalizacion
    resumen_pqrs=state.graph_summarize_pqrs
    entidades_coseno=state.graph_similitud_coseno

    client, config = get_openai_client()

    client = openai.AzureOpenAI(  # Usa AzureOpenAI en lugar de Client
    api_key="b427f4f746d24d5fa2853dc8f16979cd",
    azure_endpoint="https://oai-inboxia-poc.openai.azure.com/",
    api_version="2024-08-01-preview")

    resultados = []
    with open('/Users/rayespinoza/Visual Code/LangchainRAG/sector/Planeacion/PROMPT_SECTOR_PLANEACION_PQRS.txt', 'r', encoding='utf-8') as file: 
        PROMPT_TEMPLATE_GP = file.read()

    prompts_entidades = [
    ("004", PROMPT_TEMPLATE_GP),
    ]

    prompt = PROMPT_TEMPLATE_GP.format(
            question=query_text,
            resumen=resumen_pqrs,
            entidad_similitud=entidades_coseno)

    response = client.chat.completions.create(
                model=chat_deployment_name,
                messages=[
                    {"role": "system", "content": "Eres un experto en clasificación de entidades."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1,
                #max_tokens=100,
                logprobs=True,
                top_logprobs=5
            )

    #Obtener respuesta:
    resultado = response.choices[0].message.content.strip()

    resultados.append({
                "requirement_code": numero_query,
                "codigoEntidad": prompts_entidades[0][0],
                "resultado": int(resultado)
            })

    return {"graph_clasificacionsectorPlaneacion": resultados}


def pre_clasificacion_llm_sectores_final(state):

    df_sectorgestionpublica=pd.DataFrame(state.graph_clasificacionsectorGestionPublica)
    df_sectorgobierno=pd.DataFrame(state.graph_clasificacionsectorGobierno)
    df_sectorhacienda=pd.DataFrame(state.graph_clasificacionsectorHacienda)
    df_sectorplaneacion=pd.DataFrame(state.graph_clasificacionsectorPlaneacion)
    df_sectordesarrollo=pd.DataFrame(state.graph_clasificacionsectorDesarrollo)
    df_sectoreducacion=pd.DataFrame(state.graph_clasificacionsectorEducacion)
    df_sectorsalud=pd.DataFrame(state.graph_clasificacionsectorSalud)

    df_sectorintegracionsocial=pd.DataFrame(state.graph_clasificacionsectorIntegracionSocial)
    df_sectocultura=pd.DataFrame(state.graph_clasificacionsectorCulturaRecreacionDep)
    df_sectorambiente=pd.DataFrame(state.graph_clasificacionsectorAmbiente)
    df_sectormovilidad=pd.DataFrame(state.graph_clasificacionsectorMovilidad)
    df_sectorhabitat=pd.DataFrame(state.graph_clasificacionsectorHabitat)
    df_sectormujeres=pd.DataFrame(state.graph_clasificacionsectorMujeres)
    df_sectorseguridad=pd.DataFrame(state.graph_clasificacionsectorSeguridad)
    df_sectorgestionjuridica=pd.DataFrame(state.graph_clasificacionsectorGestionJuridica)

    clasificacion_sector_final = pd.concat([df_sectorgestionpublica, df_sectorgobierno, df_sectorhacienda, df_sectorplaneacion, df_sectordesarrollo, df_sectoreducacion, df_sectorsalud,df_sectorintegracionsocial,df_sectocultura,df_sectorambiente,df_sectormovilidad,df_sectorhabitat,df_sectormujeres,df_sectorseguridad,df_sectorgestionjuridica], ignore_index=True)

    return {"graph_clasificacionsectorFinal": clasificacion_sector_final}

def clasificacion_llm_sectores_final(state):

    df_sectorgestionpublica=pd.DataFrame(state.graph_clasificacionsectorGestionPublica)
    df_sectorgobierno=pd.DataFrame(state.graph_clasificacionsectorGobierno)
    df_sectorhacienda=pd.DataFrame(state.graph_clasificacionsectorHacienda)
    df_sectorplaneacion=pd.DataFrame(state.graph_clasificacionsectorPlaneacion)
    df_sectordesarrollo=pd.DataFrame(state.graph_clasificacionsectorDesarrollo)
    df_sectoreducacion=pd.DataFrame(state.graph_clasificacionsectorEducacion)
    df_sectorsalud=pd.DataFrame(state.graph_clasificacionsectorSalud)

    df_sectorintegracionsocial=pd.DataFrame(state.graph_clasificacionsectorIntegracionSocial)
    df_sectocultura=pd.DataFrame(state.graph_clasificacionsectorCulturaRecreacionDep)
    df_sectorambiente=pd.DataFrame(state.graph_clasificacionsectorAmbiente)
    df_sectormovilidad=pd.DataFrame(state.graph_clasificacionsectorMovilidad)
    df_sectorhabitat=pd.DataFrame(state.graph_clasificacionsectorHabitat)
    df_sectormujeres=pd.DataFrame(state.graph_clasificacionsectorMujeres)
    df_sectorseguridad=pd.DataFrame(state.graph_clasificacionsectorSeguridad)
    df_sectorgestionjuridica=pd.DataFrame(state.graph_clasificacionsectorGestionJuridica)

    clasificacion_sector_final = pd.concat([df_sectorgestionpublica, df_sectorgobierno, df_sectorhacienda, df_sectorplaneacion, df_sectordesarrollo, df_sectoreducacion, df_sectorsalud,df_sectorintegracionsocial,df_sectocultura,df_sectorambiente,df_sectormovilidad,df_sectorhabitat,df_sectormujeres,df_sectorseguridad,df_sectorgestionjuridica], ignore_index=True)

    return {"graph_clasificacionsectorFinal": clasificacion_sector_final}


def conditional_router(state):
    try:
        df_sectores = state.graph_clasificacionsectorFinal
        gobierno_activo = 1


        if gobierno_activo == 1:
            return {"graph_should_process_entities": True}
        else:
            return {"graph_should_process_entities": False}

    except Exception as e:
        print(f"Error in conditional router: {e}")
        return {"graph_should_process_entities": False}

#Agregar conexiones condicionales:
def route_to_entities(state):
       return state.graph_should_process_entities

def clasificacion_entidades_bypass(state):
    return state

def clasificacion_enviorevisionhumana(state):

    datos = {
        "requirement_code": numero_query,
        "codigoEntidad": [200, 1100, 800, 700],
        "resultado": [0, 0, 0, 0]
    }

    df_enviorevision = pd.DataFrame(datos)

    #graph_clasificacionFinal
    #return {"graph_revision_humana": df_enviorevision}
    return {"graph_clasificacionFinal": df_enviorevision}

def clasificacion_llm_entidades(state):

    df_salud = pd.DataFrame(state.graph_entidad_gobierno_pqrs)
    df_movilidad = pd.DataFrame(state.graph_entidad_movilidad_pqrs)
    df_gobierno = pd.DataFrame(state.graph_entidad_is_pqrs)
    df_integracion = pd.DataFrame(state.graph_entidad_salud_pqrs)

    clasificacion_final = pd.concat([df_salud, df_movilidad, df_gobierno, df_integracion], ignore_index=True)


    return {"graph_clasificacionFinal": clasificacion_final}


def clasificacion_inicial_llm_node5(state):

    query_text=state.graph_normalizacion
    resumen_pqrs=state.graph_summarize_pqrs
    entidades_coseno=state.graph_similitud_coseno

    client, config = get_openai_client()


    with open('/Users/rayespinoza/Visual Code/LangchainRAG/PROMPT_GOBIERNO_PQRS_EN.txt', 'r', encoding='utf-8') as file:
        PROMPT_TEMPLATE_G = file.read()

    with open('/Users/rayespinoza/Visual Code/LangchainRAG/PROMPT_MOVILIDAD_PQRS_EN.txt', 'r', encoding='utf-8') as file:
        PROMPT_TEMPLATE_M = file.read()

    with open('/Users/rayespinoza/Visual Code/LangchainRAG/PROMPT_IS_PQRS_EN.txt', 'r', encoding='utf-8') as file: 
        PROMPT_TEMPLATE_IS = file.read()

    with open('/Users/rayespinoza/Visual Code/LangchainRAG/PROMPT_SALUD_PQRS_EN.txt', 'r', encoding='utf-8') as file: 
        PROMPT_TEMPLATE_S = file.read()

    prompts_entidades = [
    ("700", PROMPT_TEMPLATE_S), #Secretaría de Salud
    ("1100", PROMPT_TEMPLATE_M), #Secretaría de Movilidad
    ("200", PROMPT_TEMPLATE_G), #Secretaría de Gobierno
    ("800", PROMPT_TEMPLATE_IS)] #Secretaría de Integración Social


    resultados = []

    for entidad, prompt_template in prompts_entidades:
        prompt = prompt_template.format(
            question=query_text,
            resumen=resumen_pqrs,
            entidad_similitud=entidades_coseno
        )

        try:
            response = client.chat.completions.create(
                model=chat_deployment_name,
                messages=[
                    {"role": "system", "content": "Eres un experto en clasificación de entidades."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1,
                #max_tokens=100,
                logprobs=True,
                top_logprobs=5
            )

            #Obtener respuesta:
            resultado = response.choices[0].message.content.strip()#response["choices"][0]["message"]["content"].strip()
            #logprobs = response.choices[0].logprobs
            #print(response.choices[0].logprobs)---->Muestra datos OK
            #print(f"{entidad}: {resultado}")

            for content_token in response.choices[0].logprobs.content:
                if content_token.token == resultado:
                    token_logprob = content_token.logprob
                    break
            probability = math.exp(token_logprob)
            #print(probability)

            #Guardar resultados:
            resultados.append({
                "requirement_code": numero_query,
                "codigoEntidad": entidad,
                "resultado": int(resultado),
                "probabilidad": probability
            })

        except Exception as e:
            #print(f"Error procesando {entidad}: {e}")
            resultados.append({
                "requirement_code": numero_query,
                "codigoEntidad": entidad,
                "Resultado": "Error",
                "probabilidad": "Error"
            })

        except openai.error.InvalidRequestError as e:
            logging.info(f"Error en la solicitud para Número de Petición {numero_query}: {e}")
            resultados.append({
                "requirement_code": numero_query,
                "codigoEntidad": "Error en la solicitud",
                "Resultado": "Error",
                "probabilidad": "Error"
            })

        except openai.error.RateLimitError:
            logging.info(f"Límite de tasa alcanzado. Esperando 60 segundos para Número de Petición {numero_query}...")
            time.sleep(60)

    #DataFrame:
    df_resultados = pd.DataFrame(resultados)

    return {"graph_clasificacionPre": df_resultados}

# Crear el grafo
graph = StateGraph(GraphState)


# In[20]:


# Añadir nodos
graph.add_node("greeting_node", greeting_node)
graph.add_node("peticion_node", peticion_node)
#graph.add_node("clasificacion_node1", clasificacion_node1)
graph.add_node("limpieza_normalizacion_node2", limpieza_normalizacion_node2)
graph.add_node("semantica_entidades_node3", semantica_entidades_node3)
graph.add_node("summarize_entidades_node4", summarize_entidades_node4)
graph.add_node("clasificacion_caso1", clasificacion_caso1)
graph.add_node("clasificacion_caso2", clasificacion_caso2)
graph.add_node("clasificacion_caso3", clasificacion_caso3)
graph.add_node("clasificacion_caso4", clasificacion_caso4)
graph.add_node("clasificacion_caso5", clasificacion_caso5)
graph.add_node("clasificacion_caso6", clasificacion_caso6)
graph.add_node("clasificacion_caso7", clasificacion_caso7)
graph.add_node("clasificacion_caso8", clasificacion_caso8)

graph.add_node("pre_clasificacion_llm_sectores_final", pre_clasificacion_llm_sectores_final)
graph.add_node("clasificacion_llm_sectores_final", clasificacion_llm_sectores_final)

#Nodo condicional:
graph.add_node("conditional_router", conditional_router)
graph.add_node("clasificacion_entidades_bypass", clasificacion_entidades_bypass)
graph.add_node("clasificacion_enviorevisionhumana", clasificacion_enviorevisionhumana)

# Añadir bordes
graph.add_edge(START, "greeting_node")
graph.add_edge("greeting_node", "peticion_node")
graph.add_edge("peticion_node", "limpieza_normalizacion_node2")
graph.add_edge("limpieza_normalizacion_node2", "semantica_entidades_node3")
#Proceso paralelo:
graph.add_edge("semantica_entidades_node3", "summarize_entidades_node4")
#graph.add_edge("summarize_entidades_node4", "clasificacion_inicial_llm_node5")
#Sectores:
graph.add_edge("summarize_entidades_node4", "clasificacion_caso1")
graph.add_edge("summarize_entidades_node4", "clasificacion_caso2")
graph.add_edge("summarize_entidades_node4", "clasificacion_caso3")
graph.add_edge("summarize_entidades_node4", "clasificacion_caso4")
graph.add_edge("summarize_entidades_node4", "clasificacion_caso5")
graph.add_edge("summarize_entidades_node4", "clasificacion_caso6")
graph.add_edge("summarize_entidades_node4", "clasificacion_caso7")
graph.add_edge("summarize_entidades_node4", "clasificacion_caso8")



#Salida Sectores
graph.add_edge("clasificacion_caso1", "pre_clasificacion_llm_sectores_final")
graph.add_edge("clasificacion_caso2", "pre_clasificacion_llm_sectores_final")
graph.add_edge("clasificacion_caso3", "pre_clasificacion_llm_sectores_final")
graph.add_edge("clasificacion_caso4", "pre_clasificacion_llm_sectores_final")
graph.add_edge("clasificacion_caso5", "pre_clasificacion_llm_sectores_final")
graph.add_edge("clasificacion_caso6", "pre_clasificacion_llm_sectores_final")
graph.add_edge("clasificacion_caso7", "pre_clasificacion_llm_sectores_final")
graph.add_edge("clasificacion_caso8", "pre_clasificacion_llm_sectores_final")

#Condicional:
graph.add_edge("pre_clasificacion_llm_sectores_final", "conditional_router")

graph.add_conditional_edges(
        "conditional_router",
        route_to_entities,
        {
            True: "clasificacion_entidades_bypass",
            False: "clasificacion_enviorevisionhumana"
        }
    )


graph.add_edge("clasificacion_entidades_bypass", "clasificacion_llm_sectores_final")


#Final:
graph.add_edge("clasificacion_llm_sectores_final", END)
#graph.add_edge("clasificacion_inicial_llm_node5", END)
graph.add_edge("clasificacion_enviorevisionhumana", END)



from IPython.display import Image, display

graph_compiled = graph.compile()

try:
    display(Image(graph_compiled.get_graph().draw_mermaid_png()))
except Exception:
    pass

#============>>>>>>>>>=======Execute=====<<<<<<<<<:

if __name__ == '__main__':
    pass  # Agrega tus funciones aquí
