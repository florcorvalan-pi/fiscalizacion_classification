import os
import spacy
from spacy.cli import download
import importlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.config import Config

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(ROOT_DIR, '.env')



_config = Config(ENV_DIR)

os.environ["PYTHONIOENCODING"] = "utf-8"

# Azure Open AI
AZURE_OPENAI_API_KEY = _config("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = _config("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = _config("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = _config("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_TEMPERATURE = _config("AZURE_OPENAI_TEMPERATURE") # Valor por defecto 0.5
AZURE_OPENAI_TOP_P = _config("AZURE_OPENAI_TOP_P") # Valor por defecto 100
AZURE_OPENAI_EMBEDDING_API_VERSION = _config("AZURE_OPENAI_EMBEDDING_API_VERSION")  
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = _config("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")      

# CosmosDB
COSMOSDB_ENDPOINT = _config("COSMOSDB_ENDPOINT")
COSMOSDB_PRIMARY_KEY = _config("COSMOSDB_PRIMARY_KEY")
COSMOSDB_DATABASE = _config("COSMOSDB_DB_NAME")
COSMOSDB_CONTAINER = _config("COSMOSDB_CONTAINER")

# Storage Account
AZURE_STORAGE_CONNECTION_STRING = _config("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = _config("AZURE_STORAGE_CONTAINER")

def create_app():
    app = FastAPI(
        title="Classificator Alcaldia de Bogota",
        description="API documentation",
        version="0.0.1",
        contact={
            "name": "Pi Data & Consulting",
            "url": "https://piconsulting.com.ar/contacto/",
            "email": "info@piconsulting.com.ar",
        })

    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app



def load_spacy_model(model_name="es_core_news_md"):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Modelo '{model_name}' no encontrado. Descargando...")
        download(model_name)
        return spacy.load(model_name)