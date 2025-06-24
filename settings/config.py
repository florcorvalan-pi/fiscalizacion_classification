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

#AI Speach project
AZURE_OPEN_AI_SPEACH=_config("AZURE_OPEN_AI_SPEACH")
AZURE_OPEN_AI_AUTOMATIZACION=_config("AZURE_OPEN_AI_AUTOMATIZACION")
AZURE_OPEN_AI_SEARCH=_config("AZURE_OPEN_AI_SEARCH")
AZURE_OPEN_AI_SEARCH_RESOLUCION=_config("AZURE_OPEN_AI_SEARCH_RESOLUCION")
AZURE_OPEN_AI_SEARCH_NORMATIVA=_config("AZURE_OPEN_AI_SEARCH_NORMATIVA")
