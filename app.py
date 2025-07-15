from flask import Flask, request, jsonify, render_template, send_from_directory
import sys
import os
import json
import logging
from pathlib import Path
import glob
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient, PartitionKey

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_cosmos_container():
    uri = os.environ.get("COSMOSDB_URI")
    key = os.environ.get("COSMOSDB_KEY")
    db_name = os.environ.get("COSMOSDB_DATABASE")
    container_name = os.environ.get("COSMOSDB_CONTAINER")
    client = CosmosClient(uri, credential=key)
    db = client.get_database_client(db_name)
    return db.get_container_client(container_name)

def get_resultado_caso(caso_id):
    """
    Busca el documento en Cosmos DB cuyo partition_id sea igual a 'caso{caso_id}'.
    """
    container = get_cosmos_container()
    partition_id = f"caso{caso_id}"
    print(f"Buscando en Cosmos: partition_id={partition_id}")
    query = "SELECT * FROM c WHERE c.partition_id = @pid"
    items = list(container.query_items(
        query=query,
        parameters=[{"name": "@pid", "value": partition_id}],
        enable_cross_partition_query=True
    ))
    print(f"Resultados encontrados: {len(items)}")
    if items:
        return items[0]
    return None

def save_resultado_caso(caso_id, data):
    container = get_cosmos_container()
    data['partition_id'] = f"caso{caso_id}"
    # Si quieres, puedes mantener el id como UUID o ponerle un id propio
    if 'id' not in data:
        import uuid
        data['id'] = str(uuid.uuid4())
    container.upsert_item(data)

def get_blob_service_client():
    account = os.environ.get("AZURE_STORAGE_ACCOUNT")
    key = os.environ.get("AZURE_STORAGE_KEY")
    return BlobServiceClient(
        f"https://{account}.blob.core.windows.net",
        credential=key
    )

def read_blob_text(blob_path, container=None):
    """
    Lee un archivo de texto desde Azure Blob Storage.
    blob_path: ruta dentro del contenedor (ej: Caso_1/aporte/caso1_aporte_1-SD_extraido.txt)
    """
    if container is None:
        container = os.environ.get("AZURE_STORAGE_CONTAINER")
    blob_service = get_blob_service_client()
    blob_client = blob_service.get_blob_client(container=container, blob=blob_path)
    stream = blob_client.download_blob()
    return stream.readall().decode("utf-8")


def objeto_a_dict(obj):
    """Convierte objetos (dataclass, etc.) a diccionarios recursivamente"""
    if hasattr(obj, '__dict__'):
        # Es un objeto con atributos
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = objeto_a_dict(value)
        return result
    elif isinstance(obj, dict):
        # Es un diccionario
        return {key: objeto_a_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Es una lista o tupla
        return [objeto_a_dict(item) for item in obj]
    else:
        # Es un tipo primitivo
        return obj

# Agregar TODOS los directorios necesarios al path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'service'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'service', 'scripts'))

# Importar el grafo desde script_modified.py
try:
    from service.scripts.demopoc import construir_grafo, EstadoCaso, SistemaAutomatizacionDO
    grafo = construir_grafo()
    sistema_do = SistemaAutomatizacionDO()
    logger.info("Grafo de script_modified.py compilado exitosamente")
except ImportError as e:
    logger.error(f"Error al importar desde script_modified.py: {e}")
    grafo = None
    sistema_do = None

app = Flask(__name__, template_folder="templates")
# Variable global para almacenar resultados (en producción usar Redis o base de datos)
resultados_cache = {}

@app.route('/')
def index():
    """Página principal con la vista del caso"""
    return render_template('view.html')

@app.route('/caso/<caso_id>')
def ver_caso(caso_id):
    """Vista específica de un caso"""
    return render_template('view.html', caso_id=caso_id)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que la API está funcionando"""
    return jsonify({
        "status": "healthy",
        "grafo_disponible": grafo is not None,
        "sistema_disponible": sistema_do is not None
    })

@app.route('/api/procesar-caso', methods=['POST'])
def procesar_caso_api():
    """
    API para procesar casos y devolver JSON para la vista
    """
    try:
        if grafo is None:
            return jsonify({"error": "Grafo no disponible"}), 500

        data = request.get_json()
        
        # Procesar según el tipo de entrada
        if 'numero_caso' in data:
            # Caso predefinido
            numero_caso = data['numero_caso']
            resultado = procesar_caso_predefinido_interno(numero_caso)
        elif 'rutas_archivos' in data:
            # Rutas específicas
            resultado = procesar_rutas_archivos_interno(data['rutas_archivos'])
        else:
            return jsonify({"error": "Datos insuficientes"}), 400

        if resultado.get('status') == 'error':
            return jsonify(resultado), 500

        # Guardar en cache para la vista
        caso_id = resultado.get('caso_id', 'default')
        resultados_cache[caso_id] = resultado

        return jsonify(resultado)

    except Exception as e:
        logger.error(f"Error en procesamiento: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/caso-data/<caso_id>')
def obtener_datos_caso(caso_id):
    """
    Devuelve el JSON real del caso si existe en Cosmos DB, si no, ejecuta el grafo, guarda el resultado y lo devuelve.
    """
    import json

    force = request.args.get('force', 'false').lower() == 'true'

    try:
        # 1. Intentar leer de Cosmos DB
        data = None
        if not force:
            try:
                data = get_resultado_caso(caso_id)
            except Exception as e:
                logger.error(f"Error al consultar Cosmos DB: {e}")
                return {"error": f"Error al consultar Cosmos DB: {str(e)}"}, 500

        if data is None or force:
            # Si no existe, ejecuta el grafo y guarda el resultado
            from service.scripts.script_modified import construir_grafo, SistemaAutomatizacionDO
            grafo = construir_grafo()
            sistema_do = SistemaAutomatizacionDO()

            # Aquí deberías leer los archivos de Blob Storage, no del disco local
            # (esto ya lo vimos en la respuesta anterior)
            # Ejemplo:
            corrida_path = f"Caso_{caso_id}/corrida/caso{caso_id}_corrida_de_vista-SD_extraido.txt"
            contenido_corrida = read_blob_text(corrida_path)

            aportes = []
            for i in range(1, 7):
                aporte_path = f"Caso_{caso_id}/aporte/caso{caso_id}_aporte_{i}-SD_extraido.txt"
                try:
                    aportes.append(read_blob_text(aporte_path))
                except Exception:
                    pass

            if not contenido_corrida or not aportes:
                return {
                    "expediente": f"DO-2025-{str(caso_id).zfill(6)}",
                    "contribuyente": None,
                    "vencimiento": None,
                    "estado": "Error",
                    "historial": [],
                    "documentos": [],
                    "analisisIA": None,
                    "resolucion": None,
                    "error": "Faltan archivos en Blob Storage"
                }, 404

            estado_inicial = {
                "texto_corrida": contenido_corrida,
                "textos_descargo": aportes
            }
            resultado_grafo = grafo.invoke(estado_inicial)
            # Convierte objetos no serializables a dict
            resultado_serializable = {}
            for key, value in resultado_grafo.items():
                if hasattr(value, '__dict__'):
                    resultado_serializable[key] = value.__dict__
                else:
                    resultado_serializable[key] = value

            # Guarda el resultado en Cosmos DB
            save_resultado_caso(caso_id, resultado_serializable)
            data = resultado_serializable

        # Agrega los campos que espera el frontend si hace falta
        data['expediente'] = f"DO-2025-{str(caso_id).zfill(6)}"
        data['estado'] = "En Análisis"
        data['vencimiento'] = "2025-07-25"
        data['documentos'] = obtener_documentos_caso(caso_id)

        if 'historial' not in data:
            data['historial'] = [
                {
                    "fecha": "2025-01-10",
                    "titulo": "Comunicación de Inicio",
                    "descripcion": "Notificación de inicio de inspección enviada al DFE",
                    "completado": True
                },
                # ...otros eventos si quieres...
            ]
        if 'documentos' not in data:
            data['documentos'] = []
        return data
    except Exception as e:
        return {"error": str(e)}, 500
    
def procesar_caso_predefinido_interno(numero_caso):
    """Función interna para procesar casos predefinidos"""
    try:
        base_path = os.path.join(PROJECT_ROOT, "data", "casos", "extract")
        
        rutas_archivos = {
            "corrida": os.path.join(base_path, f"Caso_{numero_caso}", "corrida", f"caso{numero_caso}_corrida_de_vista-SD_extraido.txt"),
            "aportes": [
                os.path.join(base_path, f"Caso_{numero_caso}", "aporte", f"caso{numero_caso}_aporte_{i}-SD_extraido.txt")
                for i in range(1, 7)  # Buscar hasta 6 aportes
            ]
        }

        return procesar_rutas_archivos_interno(rutas_archivos, numero_caso)

    except Exception as e:
        return {"status": "error", "mensaje": str(e)}

def procesar_rutas_archivos_interno(rutas_archivos, caso_id=None):
    """Función interna para procesar archivos por rutas"""
    try:
        # Leer archivo de corrida
        if not os.path.exists(rutas_archivos["corrida"]):
            return {"status": "error", "mensaje": "Archivo de corrida no encontrado"}
        
        with open(rutas_archivos["corrida"], "r", encoding="utf-8") as f:
            contenido_corrida = f.read()
        
        # Leer archivos de aportes
        contenidos_aportes = []
        for ruta in rutas_archivos["aportes"]:
            if os.path.exists(ruta):
                with open(ruta, "r", encoding="utf-8") as f:
                    contenidos_aportes.append(f.read())

        if not contenidos_aportes:
            return {"status": "error", "mensaje": "No se encontraron archivos de aportes"}

        # Estado inicial para el grafo
        estado_inicial = {
            "texto_corrida": contenido_corrida,
            "textos_descargo": contenidos_aportes
        }

        # Ejecutar grafo
        resultado_grafo = grafo.invoke(estado_inicial)
        
        # Convertir objetos a diccionarios
        resultado_serializable = objeto_a_dict(resultado_grafo)
        
        return {
            "status": "success",
            "caso_id": caso_id or "procesado",
            "resultado": resultado_serializable,
            "archivos_procesados": {
                "corrida": rutas_archivos["corrida"],
                "aportes_encontrados": len(contenidos_aportes)
            }
        }

    except Exception as e:
        logger.error(f"Error en procesamiento interno: {str(e)}")
        return {"status": "error", "mensaje": str(e)}


def transformar_para_vista(resultado_procesamiento, caso_id):
    # Toma todo el resultado original
    datos = dict(resultado_procesamiento.get('resultado', {}))
    # Agrega los campos que espera la vista (si no están)
    datos['expediente'] = f"DO-2025-{str(caso_id).zfill(6)}"
    datos['estado'] = "En Análisis"
    datos['vencimiento'] = calcular_vencimiento(datos.get('plazo_para_descargo', 15))
    # Si quieres, puedes agregar el resultado_procesamiento completo para debugging
    datos['debug'] = resultado_procesamiento
    datos['historial'] = [
        {
            "fecha": "2025-01-10",
            "titulo": "Comunicación de Inicio",
            "descripcion": "Notificación de inicio de inspección enviada al DFE",
            "completado": True
        },
        {
            "fecha": "2025-01-25",
            "titulo": "Cierre - Pase a DO",
            "descripcion": "Derivación a Determinación de Oficio",
            "completado": True
        },
        {
            "fecha": "2025-02-15",
            "titulo": "Corrida de Vista",
            "descripcion": "Notificación de corrida de vista enviada",
            "completado": True
        },
        {
            "fecha": "2025-03-01",
            "titulo": "Descargo del Contribuyente",
            "descripcion": "Recepción de descargo y documentación",
            "completado": True
        },
        {
            "fecha": "2025-03-10",
            "titulo": "Análisis en Curso",
            "descripcion": "Análisis de documentos y generación de resolución",
            "completado": False
        }
    ]

    # Aquí agregas los documentos reales
    datos['documentos'] = obtener_documentos_caso(caso_id)

    return datos
def safe_get_list_length(obj, key, default=0):
    """Obtiene la longitud de una lista de forma segura"""
    try:
        value = obj.get(key, [])
        if isinstance(value, (list, tuple)):
            return len(value)
        elif isinstance(value, int):
            return value
        else:
            return default
    except:
        return default

def safe_get_value(obj, key, default=None):
    """Obtiene un valor de forma segura"""
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        elif hasattr(obj, key):
            return getattr(obj, key, default)
        else:
            return default
    except:
        return default

def generar_historial_seguro(datos_corrida, resultado_procesamiento):
    """Genera historial basado en los datos reales del caso - versión segura"""
    try:
        # Obtener valores de forma segura
        tributo = safe_get_value(datos_corrida, 'tributo', 'N/A')
        periodos = safe_get_value(datos_corrida, 'periodos_fiscales', [])
        
        # Asegurar que periodos sea una lista
        if not isinstance(periodos, list):
            periodos = [str(periodos)] if periodos else []
        
        # Obtener cantidad de aportes de forma segura
        archivos_procesados = resultado_procesamiento.get('archivos_procesados', {})
        aportes_count = archivos_procesados.get('aportes_encontrados', 0)
        
        # Asegurar que aportes_count sea un número
        if not isinstance(aportes_count, int):
            try:
                aportes_count = int(aportes_count)
            except:
                aportes_count = 0

        historial = [
            {
                "fecha": "2025-01-10",
                "titulo": "Comunicación de Inicio",
                "descripcion": "Notificación de inicio de inspección enviada al DFE",
                "completado": True
            },
            {
                "fecha": "2025-01-25",
                "titulo": "Cierre - Pase a DO",
                "descripcion": "Derivación a Determinación de Oficio",
                "completado": True
            },
            {
                "fecha": "2025-02-15",
                "titulo": "Corrida de Vista",
                "descripcion": f"Tributo: {tributo} - Períodos: {', '.join(periodos) if periodos else 'N/A'}",
                "completado": True
            },
            {
                "fecha": "2025-03-01",
                "titulo": "Descargo del Contribuyente",
                "descripcion": f"Recepción de {aportes_count} documentos",
                "completado": True
            },
            {
                "fecha": "2025-03-10",
                "titulo": "Análisis Completado",
                "descripcion": f"Análisis IA completado - Estado: Procesado",
                "completado": True
            }
        ]
        
        return historial
        
    except Exception as e:
        logger.error(f"Error generando historial: {str(e)}")
        return [
            {
                "fecha": "2025-03-10",
                "titulo": "Análisis Completado",
                "descripcion": "Procesamiento completado con datos limitados",
                "completado": True
            }
        ]

def generar_analisis_ia_seguro(clasificacion, hechos, validacion):
    """Genera análisis IA basado en los datos reales - versión segura"""
    try:
        # Calcular confianza basada en los datos
        confianza = 50  # Base
        
        # Aumentar confianza según factores
        clasificaciones = clasificacion.get('clasificaciones', []) if isinstance(clasificacion, dict) else []
        if isinstance(clasificaciones, list):
            confianza += len(clasificaciones) * 5
        
        suficiencia = safe_get_value(hechos, 'suficiencia_probatoria', 'parcial')
        if suficiencia == 'suficiente':
            confianza += 20
        elif suficiencia == 'parcial':
            confianza += 10
            
        cumplimiento = safe_get_value(validacion, 'cumplimiento_general', 'parcial')
        if cumplimiento == 'cumple':
            confianza += 15
        elif cumplimiento == 'parcial':
            confianza += 8
            
        confianza = min(confianza, 95)  # Máximo 95%
        
        # Determinar completitud
        completitud_map = {
            'cumple': 'Completo',
            'parcial': 'Parcial',
            'no_cumple': 'Insuficiente'
        }
        completitud = completitud_map.get(cumplimiento, 'Parcial')
        
        # Generar observaciones
        observaciones_parts = []
        
        if isinstance(clasificaciones, list) and len(clasificaciones) > 0:
            observaciones_parts.append(f"Clasificaciones detectadas: {len(clasificaciones)}")
            
        hechos_lista = safe_get_value(hechos, 'hechos', [])
        if isinstance(hechos_lista, list) and len(hechos_lista) > 0:
            observaciones_parts.append(f"Hechos analizados: {len(hechos_lista)}")
            
        requisitos = safe_get_value(validacion, 'detalles_por_requisito', [])
        if isinstance(requisitos, list) and len(requisitos) > 0:
            observaciones_parts.append(f"Requisitos evaluados: {len(requisitos)}")
            
        observaciones = '; '.join(observaciones_parts) if observaciones_parts else 'Análisis completado'
        
        recomendacion = safe_get_value(validacion, 'recomendacion_resolucion', 'Revisar documentación')
        
        return {
            "confianza": confianza,
            "completitud": completitud,
            "observaciones": observaciones,
            "recomendacion": recomendacion
        }
        
    except Exception as e:
        logger.error(f"Error generando análisis IA: {str(e)}")
        return {
            "confianza": 75,
            "completitud": "Parcial",
            "observaciones": "Error en análisis automático",
            "recomendacion": "Revisar manualmente"
        }

def generar_lista_documentos_seguro(resultado_procesamiento):
    """Genera lista de documentos basada en el procesamiento - versión segura"""
    try:
        documentos = []
        
        # Documento de corrida
        documentos.append({
            "nombre": "Corrida de Vista",
            "tipo": "TXT",
            "tamaño": "2.1 MB",
            "fecha": "2025-02-15"
        })
        
        # Documentos de aportes
        archivos_procesados = resultado_procesamiento.get('archivos_procesados', {})
        aportes_count = archivos_procesados.get('aportes_encontrados', 0)
        
        # Asegurar que aportes_count sea un número
        if not isinstance(aportes_count, int):
            try:
                aportes_count = int(aportes_count)
            except:
                aportes_count = 0
        
        for i in range(1, aportes_count + 1):
            documentos.append({
                "nombre": f"Aporte del Contribuyente {i}",
                "tipo": "TXT",
                "tamaño": f"{1.2 + (i * 0.3):.1f} MB",
                "fecha": "2025-03-01"
            })
        
        return documentos
        
    except Exception as e:
        logger.error(f"Error generando lista de documentos: {str(e)}")
        return [
            {
                "nombre": "Corrida de Vista",
                "tipo": "TXT",
                "tamaño": "2.1 MB",
                "fecha": "2025-02-15"
            }
        ]

def calcular_vencimiento(plazo_dias):
    """Calcula fecha de vencimiento basada en el plazo"""
    from datetime import datetime, timedelta
    try:
        # Asegurar que plazo_dias sea un número
        if not isinstance(plazo_dias, int):
            try:
                plazo_dias = int(plazo_dias)
            except:
                plazo_dias = 15  # Default
                
        fecha_base = datetime.now()
        vencimiento = fecha_base + timedelta(days=plazo_dias)
        return vencimiento.strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"Error calculando vencimiento: {str(e)}")
        return "2025-03-20"  # Fecha por defecto

import os

def obtener_documentos_caso(numero_caso):
    """
    Lista los documentos PDF (o los que quieras) de un caso desde Azure Blob Storage,
    buscando en la raíz de la carpeta del caso.
    """
    documentos = []
    container = os.environ.get("AZURE_STORAGE_CONTAINER")
    blob_service = get_blob_service_client()
    carpeta_caso = f"Caso_{numero_caso}/"
    extensiones_validas = ['.pdf', '.PDF', '.txt']  # Agrega .txt si quieres mostrar esos también

    blobs = blob_service.get_container_client(container).list_blobs(name_starts_with=carpeta_caso)
    for blob in blobs:
        # Solo archivos en la raíz de la carpeta del caso (no en subcarpetas)
        nombre_relativo = blob.name[len(carpeta_caso):]
        if '/' in nombre_relativo:
            continue  # Está en una subcarpeta, lo ignoramos
        if any(blob.name.lower().endswith(ext) for ext in extensiones_validas):
            documentos.append({
                "nombre": os.path.basename(blob.name),
                "tipo": blob.name.split('.')[-1].upper(),
                "tamaño": f"{blob.size // 1024} KB",
                "fecha": str(blob.last_modified)[:10] if blob.last_modified else ""
            })
    print(f"Total documentos encontrados en Blob (raíz de Caso_{numero_caso}): {len(documentos)}")
    return documentos

@app.route('/api/test-data/<caso_id>')
def test_data(caso_id):
    """Endpoint simple para probar qué datos se están devolviendo"""
    try:
        # Datos de prueba simples
        datos_test = {
            "expediente": f"DO-2025-{str(caso_id).zfill(6)}",
            "contribuyente": {
                "nombre": "EMPRESA TEST S.A.",
                "cuit": "30-12345678-9"
            },
            "vencimiento": "2025-03-20",
            "estado": "En Análisis",
            "historial": [
                {
                    "fecha": "2025-01-10",
                    "titulo": "Test - Comunicación de Inicio",
                    "descripcion": "Datos de prueba",
                    "completado": True
                },
                {
                    "fecha": "2025-02-15",
                    "titulo": "Test - Corrida de Vista",
                    "descripcion": "Datos de prueba",
                    "completado": True
                }
            ],
            "documentos": [
                {
                    "nombre": "Documento Test",
                    "tipo": "TXT",
                    "tamaño": "1.0 MB",
                    "fecha": "2025-02-15"
                }
            ],
            "analisisIA": {
                "confianza": 85,
                "completitud": "Completo",
                "observaciones": "Datos de prueba",
                "recomendacion": "Test OK"
            },
            "resolucion": "Esta es una resolución de prueba para verificar que los datos se cargan correctamente.",
            "debug": True
        }
        
        return jsonify(datos_test)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/documentos/caso/<caso_id>/<path:nombre_archivo>')
def servir_documento(caso_id, nombre_archivo):
    """
    Sirve un documento real del caso para ver o descargar.
    """
    base_path = os.path.join(PROJECT_ROOT, "data", "casos", f"Caso_{caso_id}")
    print(f"Buscando archivo: {nombre_archivo}")  # Debug
    print(f"En carpeta: {base_path}")  # Debug
    
    # Buscar en subcarpetas relevantes
    subcarpetas = [
        "1-Corrida de Vista",
        "2-Aporte del contribuyente",
        "3-Resolución Determinativa",
        "3- Resolución",
        "3-Resolucion Determinativa"
    ]
    for sub in subcarpetas:
        carpeta = os.path.join(base_path, sub)
        archivo_path = os.path.join(carpeta, nombre_archivo)
        print(f"Probando: {archivo_path}")  # Debug
        print(f"¿Existe? {os.path.exists(archivo_path)}")  # Debug
        if os.path.exists(archivo_path):
            print(f"Archivo encontrado en: {carpeta}")  # Debug
            return send_from_directory(carpeta, nombre_archivo, as_attachment=False)
    
    print(f"Archivo NO encontrado: {nombre_archivo}")  # Debug
    return f"Archivo no encontrado: {nombre_archivo}", 404

@app.route('/api/casos')
def obtener_casos_dashboard():
    """
    Endpoint para obtener todos los casos reales para el dashboard
    """
    try:
        casos = []
        
        # Buscar casos reales en /data/casos/
        casos_path = os.path.join(PROJECT_ROOT, "data", "casos")
        
        if not os.path.exists(casos_path):
            logger.error(f"No existe la carpeta de casos: {casos_path}")
            return jsonify({"error": "Carpeta de casos no encontrada"}), 404
        
        # Listar todas las carpetas de casos (Caso_1, Caso_2, etc.)
        for item in os.listdir(casos_path):
            if item.startswith("Caso_") and os.path.isdir(os.path.join(casos_path, item)):
                try:
                    # Extraer número de caso
                    numero_caso = item.split("_")[1]
                    caso_id = int(numero_caso)
                    
                    # Verificar que el caso tenga contenido
                    caso_path = os.path.join(casos_path, item)
                    
                    # Contar documentos reales
                    documentos_count = 0
                    subcarpetas = ["1-Corrida de Vista", "2-Aporte del contribuyente", "3-Resolución Determinativa", "3- Resolución", "3-Resolucion Determinativa"]
                    
                    for subcarpeta in subcarpetas:
                        subcarpeta_path = os.path.join(caso_path, subcarpeta)
                        if os.path.exists(subcarpeta_path):
                            for archivo in os.listdir(subcarpeta_path):
                                if os.path.isfile(os.path.join(subcarpeta_path, archivo)):
                                    documentos_count += 1
                    
                    # Solo incluir casos que tengan documentos
                    if documentos_count > 0:
                        # Calcular días restantes basado en el número de caso (para simular diferentes vencimientos)
                        dias_restantes = max(1, 15 - (caso_id * 1.5))
                        
                        # Determinar estado basado en días restantes
                        if dias_restantes <= 2:
                            estado = "vencido"
                            prioridad = "alta"
                        elif dias_restantes <= 7:
                            estado = "proceso"
                            prioridad = "media"
                        else:
                            estado = "pendiente"
                            prioridad = "baja"
                        
                        # Determinar etapa basada en el contenido de carpetas
                        etapa = "Corrida de Vista"  # Default
                        if os.path.exists(os.path.join(caso_path, "3-Resolución Determinativa")) or \
                           os.path.exists(os.path.join(caso_path, "3- Resolución")) or \
                           os.path.exists(os.path.join(caso_path, "3-Resolucion Determinativa")):
                            etapa = "Resolución Determinativa"
                        elif os.path.exists(os.path.join(caso_path, "2-Aporte del contribuyente")):
                            etapa = "Análisis de Descargo"
                        
                        casos.append({
                            "expediente": f"DO-2025-{str(caso_id).zfill(6)}",
                            "contribuyente": f"CONTRIBUYENTE CASO{caso_id}",
                            "cuit": f"11-{str(11111111 + caso_id).zfill(8)}-{caso_id}",
                            "etapa": etapa,
                            "estado": estado,
                            "prioridad": prioridad,
                            "vencimiento": calcular_fecha_vencimiento(dias_restantes),
                            "diasRestantes": int(dias_restantes),
                            "fechaInicio": "2025-01-10",
                            "fechaCompletado": None if estado != "completado" else "2025-03-15",
                            "analisisIA": True if caso_id % 2 == 0 else False,
                            "documentosCount": documentos_count
                        })
                        
                        logger.info(f"Caso {caso_id} agregado con {documentos_count} documentos")
                        
                except Exception as e:
                    logger.error(f"Error procesando caso {item}: {str(e)}")
                    continue
        
        # Ordenar casos por número
        casos.sort(key=lambda x: int(x["expediente"].split("-")[2]))
        
        # Calcular métricas reales
        total_casos = len(casos)
        casos_criticos = len([c for c in casos if c["diasRestantes"] <= 2])
        casos_advertencia = len([c for c in casos if c["diasRestantes"] > 2 and c["diasRestantes"] <= 7])
        completados_hoy = len([c for c in casos if c["estado"] == "completado"])
        tiempo_promedio = 8.5  # Simulado
        procesados_ia = len([c for c in casos if c["analisisIA"]])
        
        # Generar alertas basadas en casos reales
        alertas = []
        for caso in casos:
            if caso["diasRestantes"] <= 2:
                alertas.append({
                    "id": f"critical-{caso['expediente']}",
                    "type": "critical",
                    "title": "Vencimiento Crítico",
                    "description": f"{caso['expediente']} vence en {caso['diasRestantes']} días",
                    "time": f"{caso['diasRestantes']} días",
                    "caseId": caso['expediente']
                })
            elif caso["diasRestantes"] <= 7:
                alertas.append({
                    "id": f"warning-{caso['expediente']}",
                    "type": "warning",
                    "title": "Requiere Atención",
                    "description": f"{caso['expediente']} vence en {caso['diasRestantes']} días",
                    "time": f"{caso['diasRestantes']} días",
                    "caseId": caso['expediente']
                })
        
        logger.info(f"Dashboard: {total_casos} casos cargados, {len(alertas)} alertas generadas")
        
        return jsonify({
            "cases": casos,
            "metrics": {
                "totalCases": total_casos,
                "criticalCases": casos_criticos,
                "warningCases": casos_advertencia,
                "completedToday": completados_hoy,
                "avgTime": tiempo_promedio,
                "aiProcessed": procesados_ia
            },
            "alerts": alertas
        })
        
    except Exception as e:
        logger.error(f"Error en obtener_casos_dashboard: {str(e)}")
        return jsonify({"error": str(e)}), 500

def calcular_fecha_vencimiento(dias_restantes):
    """Calcula fecha de vencimiento basada en días restantes"""
    from datetime import datetime, timedelta
    fecha_base = datetime.now()
    vencimiento = fecha_base + timedelta(days=dias_restantes)
    return vencimiento.strftime('%Y-%m-%d')

@app.route('/dashboard')
def dashboard():
    """Vista del dashboard principal"""
    return render_template('dashboard.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
