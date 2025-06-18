import logging
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from apps.core.schemas import RunSchema
from service.classifier_service import Classifier
from service.azure_cosmosdb_services import CosmosDB
from settings.config import load_spacy_model

router = APIRouter()


@router.get('/ping', status_code=status.HTTP_200_OK)
def ping():
    try:
        return JSONResponse(
            content={'message': 'pong'}, 
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        raise JSONResponse(
            content={'message': f'Error in ping: {str(e)}'},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
@router.post('/run', status_code=status.HTTP_200_OK)
def run(req: RunSchema) -> dict:

    try:

        # Paso 1: Obtener clasificación
        # ========================================================
        cs = Classifier()
        answer, db_answer = cs.main(req.id)

        if not answer:
            raise ValueError(f"No se obtuvo una respuesta válida para el ID: {id}")

        # Paso 2: Registrar en la base de datos
        # ========================================================
        cdb = CosmosDB()
        for dba in db_answer:
            cdb.update_item(dba)

        # Paso 3: Registrar en el Servicio de Alcaldía        
        # ========================================================

        # # Manejar casos donde 'codigoEntidad' sea NaN
        # codigo_entidad = answer.get("codigoEntidad")
        # if codigo_entidad is None or (isinstance(codigo_entidad, str) and codigo_entidad.lower() == "nan"):
        #     codigo_entidad = "No disponible"
        #     raise ValueError(f"El código de entidad no está disponible para el ID: {id}")

        # else:
        #     # ========================================================
        #     update_requirement_url = 'https://fa-inboxia-poc.azurewebsites.net/api/update_requirement?code=1Qa2gohYc95PdHaLaiQdyyZFxRwX4SHxuod1vUUYS_tyAzFuWmkoDA%3D%3D'
        #     body = {
        #         "id": id,
        #         "entities": [codigo_entidad]
        #     }
        #     res = requests.post(update_requirement_url, json=body, timeout=120)
        #     if res.status_code != 200:
        #         response['client_service_update'] = f'Error durante la actualización del registro {res.status_code} - {res.text}'
        #     else:
        #         response['client_service_update'] = f'Actualización de datos exitosa {res.status_code} - {res.text}'

        return JSONResponse(
            content={'classification': answer},
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        logging.error(f"ERROR: {str(e)}")
        return JSONResponse(
            content=f"Error processing request. Detail: {str(e)}", 
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
@router.get('/download_model', status_code=status.HTTP_200_OK)
def download_model():
    try:
        load_spacy_model()
        return JSONResponse(
            content={'message': 'Model downloaded successfully'},
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        return JSONResponse(
            content={'message': f'Error downdloading model: {str(e)}'}
        )