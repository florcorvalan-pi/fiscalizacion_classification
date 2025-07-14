import json
import logging
from datetime import datetime
from typing import Dict, List, TypedDict
from dataclasses import dataclass

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os

# LangGraph
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
# ---------------------------------------------------------------------------
# Dataclasses de dominio

@dataclass
class DatosCorridaVista:
    """Estructura de datos extraídos de la corrida de vista"""
    contribuyente: str
    cuit: str
    tributo: str
    periodos_fiscales: List[str]
    actividad_codificada: str
    normativa_aplicada: List[str]
    motivo_del_ajuste: str
    infraccion_detectada: str
    multa_estimativa: str
    recargos_resarcitorios: str
    plazo_para_descargo: int
    canal_presentacion: str
    requisitos_para_justificar: List[str]
    documentos_adicionales_mencionados: List[str]


@dataclass
class ResultadoValidacion:
    """Resultado de la validación de cumplimiento"""
    cumplimiento_general: str  # "cumple", "parcial", "no_cumple"
    detalles_por_requisito: List[Dict]
    recomendacion_resolucion: str
    fundamentos_juridicos: List[str]

import re

def limpiar_json_response(raw_content: str) -> str:
    """Limpia y corrige problemas comunes en respuestas JSON de OpenAI"""
    
    # Eliminar markdown si existe
    if raw_content.startswith("```json"):
        json_string = raw_content[len("```json"):].strip()
        if json_string.endswith("```"):
            json_string = json_string[:-len("```")].strip()
    else:
        json_string = raw_content
    
    # Limpiar comas extras antes de } y ]
    json_string = re.sub(r',\s*}', '}', json_string)
    json_string = re.sub(r',\s*]', ']', json_string)
    
    # Limpiar espacios y saltos de línea problemáticos
    json_string = json_string.strip()
    
    return json_string

class SistemaAutomatizacionDO:
    """Sistema principal para automatización de Determinación de Oficio"""

    def __init__(self):
        # Configuración Azure OpenAI
        self.openai_client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        )

        # Configuración Azure Search
        self.search_client = SearchClient(
            endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
            index_name=os.environ.get("AZURE_SEARCH_INDEX_CBAANALISIS"),
            credential=AzureKeyCredential(os.environ.get("AZURE_SEARCH_KEY_CBAANALISIS")),
        )
        self.norma_clients = {
            "codigo_tributario": SearchClient(
                endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
                index_name=os.environ.get("AZURE_SEARCH_INDEX_CODIGO_TRIBUTARIO"),
                credential=AzureKeyCredential(os.environ.get("AZURE_SEARCH_KEY_CODIGO_TRIBUTARIO")),
            ),
            "jurisprudencia": SearchClient(
                endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
                index_name=os.environ.get("AZURE_SEARCH_INDEX_JURISPRUDENCIA"),
                credential=AzureKeyCredential(os.environ.get("AZURE_SEARCH_KEY_JURISPRUDENCIA")),
            ),
            "resolucion_normativa": SearchClient(
                endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT_RESOLUCION"),
                index_name=os.environ.get("AZURE_SEARCH_INDEX_RESOLUCION_NORMATIVA"),
                credential=AzureKeyCredential(os.environ.get("AZURE_SEARCH_KEY_RESOLUCION")),
            ),
            "decreto_reglamentario": SearchClient(
                endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT_RESOLUCION"),
                index_name=os.environ.get("AZURE_SEARCH_INDEX_DECRETO_REGLAMENTARIO"),
                credential=AzureKeyCredential(os.environ.get("AZURE_SEARCH_KEY_DECRETO")),
            ),
        }

        # Cargar base de conocimiento normativo
        self.normativa_base = self._cargar_normativa_base()


    def _cargar_normativa_base(self) -> Dict:
        """Carga la base de conocimiento normativo"""
        return {
            "codigo_tributario": {
                "ley": "Ley Provincial N° 6006",
                "articulos_clave": ["Art. 12", "Art. 15", "Art. 67", "Art. 89"],
                "infracciones": {
                    "omision": "Art. 67",
                    "defraudacion": "Art. 89",
                    "resistencia": "Art. 95",
                },
            },
            "procedimiento_fiscal": {
                "plazos": {"descargo": 15, "recurso": 15, "reconsideracion": 30},
                "notificaciones": "DFE obligatorio",
            },
            "principios_constitucionales": [
                "Legalidad (Art. 19 C.N.)",
                "Propiedad - No confiscatoriedad (Art. 17 C.N.)",
                "Igualdad (Art. 16 C.N.)",
                "Debido proceso adjetivo",
            ],
        }
        
    def extraer_datos_corrida_vista(self, texto: str) -> DatosCorridaVista:
        """Extrae y estructura datos de la corrida de vista con validación normativa"""
        
        
        with open("service/prompt/automization/prompt_corrida_vista.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        
        prompt = prompt_template.format(texto_pqrs=texto)
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1
            )
            print(response.choices[0].message.content.strip())
            raw_response_content = response.choices[0].message.content.strip()
            json_string = limpiar_json_response(raw_response_content)
            
            datos_json = json.loads(json_string)
            print(datos_json)
            
            return DatosCorridaVista(
                contribuyente=datos_json.get('contribuyente', ''),
                cuit=datos_json.get('cuit', ''),
                tributo=datos_json.get('tributo', ''),
                periodos_fiscales=datos_json.get('periodos_fiscales', []),
                actividad_codificada=datos_json.get('actividad_codificada', ''),
                normativa_aplicada=datos_json.get('normativa_aplicada', []),
                motivo_del_ajuste=datos_json.get('motivo_del_ajuste', ''),
                infraccion_detectada=datos_json.get('infraccion_detectada', ''),
                multa_estimativa=datos_json.get('multa_estimativa', ''),
                recargos_resarcitorios=datos_json.get('recargos_resarcitorios', ''),
                plazo_para_descargo=datos_json.get('plazo_para_descargo', 15),
                canal_presentacion=datos_json.get('canal_presentacion', ''),
                requisitos_para_justificar=datos_json.get('requisitos_para_justificar', []),
                documentos_adicionales_mencionados=datos_json.get('documentos_adicionales_mencionados', [])
            )
            
        except Exception as e:
            logger.error(f"Error al extraer datos de corrida de vista: {e}")
            raise

    def clasificar_impugnacion_avanzada(self, textos_descargo: List[str]) -> Dict:
        """Clasificación mejorada con análisis jurídico profundo"""
        
        texto_completo = " ".join(textos_descargo)
        
        prompt = f"""
        Eres un magistrado tributario experto en la Provincia de Córdoba.
        
        MARCO NORMATIVO APLICABLE:
        {json.dumps(self.normativa_base, indent=2, ensure_ascii=False)}
        
        TAXONOMÍA DE IMPUGNACIONES (usar exactamente estas categorías):
        
        1. Procedimiento de DO:
           • Ausencia de fiscalización
           • Nulidad por violación al debido proceso adjetivo
           • Impugnación de la notificación al DFE
        
        2. Afectación a principios constitucionales:
           • Legalidad (Art. 19 C.N.)
           • Propiedad – Confiscatoriedad (Art. 17 C.N.)
           • Igualdad (Art. 16 C.N.)
           • Inconstitucionalidad de las normas del C.T.P.
        
        3. Ajuste fiscal:
           • Ventas de bienes de uso
           • Operaciones no gravadas o exentas
           • Recupero de gastos
           • Bonificaciones y notas de crédito/débito
           • Actividad industrial
           • Actividad primaria exenta
        
        4. Estimación del impuesto:
           • Convenio Multilateral: coeficiente unificado
           • Atribución de diferencias
        
        5. Recargos resarcitorios
        
        6. Multa por omisión:
           • Principios del derecho penal
           • Falta de elementos objetivo/subjetivo
           • Error excusable
        
        7. Prueba aportada:
           • Documental
           • Pericial
           • Informativa
           • No aporta
        
        ANÁLISIS REQUERIDO:
        1. Clasifica cada argumento según la taxonomía
        2. Evalúa la solidez jurídica de cada impugnación
        3. Identifica contradicciones o inconsistencias
        4. Determina si los argumentos tienen sustento normativo
        
        ## FORMATO DE SALIDA ESPERADO (responder solamente el json)
        Formato de respuesta JSON:
        {{
            "clasificaciones": [
                {{
                    "tipo": "string",
                    "subtipo": "string",
                    "evidencia_textual": "string",
                    "solidez_juridica": "alta|media|baja",
                    "fundamento_normativo": "artículo/norma aplicable",
                    "observaciones": "análisis crítico"
                }}
            ],
            "resumen_estrategia_defensa": "string",
            "puntos_debiles_impugnacion": ["string"],
            "recomendacion_respuesta": "string"
        }}
        
        TEXTO DEL DESCARGO:
        {texto_completo}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1
            )
            
            raw_response_content = response.choices[0].message.content.strip()
            # Elimina las comillas triples y la palabra "json" si existen
            if raw_response_content.startswith("```json"):
                json_string = raw_response_content[len("```json"):].strip()
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].strip()
            else:
                json_string = raw_response_content
                
            print(json_string)
            
            
            return json.loads(json_string)
            
        except Exception as e:
            logger.error(f"Error en clasificación avanzada: {e}")
            raise

    def extraer_hechos_estructurados(self, textos_descargo: List[str], texto_corrida: str) -> Dict:
        """Extrae hechos con análisis de relevancia jurídica"""
        
        texto_completo = " ".join(textos_descargo)
        
        with open("service/prompt/automization/prompt_hechos_estructurado.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        
        prompt = prompt_template.format(
            texto_corrida=texto_corrida,
            texto_completo=texto_completo)
        
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1
            )
            
            raw_response_content = response.choices[0].message.content.strip()
            # Elimina las comillas triples y la palabra "json" si existen
            json_string = limpiar_json_response(raw_response_content)
            
            try:
                return json.loads(json_string)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON: {e}")
                logger.error(f"JSON problemático: {json_string}")
            
        except Exception as e:
            logger.error(f"Error en extracción de hechos: {e}")
            raise

    def buscar_jurisprudencia_contextual(self, clasificacion: Dict, hechos: Dict) -> Dict:
        """Búsqueda de jurisprudencia contextualizada y relevante"""
        
        # Construir query semántica basada en clasificación y hechos
        queries = []
        
        for clasificacion_item in clasificacion.get('clasificaciones', []):
            tipo = clasificacion_item.get('tipo', '')
            subtipo = clasificacion_item.get('subtipo', '')
            queries.append(f"{tipo} {subtipo}")
        
        # Agregar hechos relevantes para búsqueda
        hechos_relevantes = [
            h['descripcion'] for h in hechos.get('hechos', [])
            if h.get('relevancia_juridica') == 'alta'
        ]
        
        jurisprudencia_encontrada = []
        
        for query in queries[:3]:  # Limitar búsquedas
            try:
                results = self.search_client.search(
                    search_text=query,
                    top=3
                   
                    
                )
                
                for result in results:
                    jurisprudencia_encontrada.append({
                        'tipo': result.get('tipo', ''),
                        'agravio': result.get('agravio', ''),
                        'respuesta': result.get('respuesta', '')[:500],  # Limitar texto
                        'relevancia': result.get('@search.score', 0),
                        'query_origen': query
                    })
                    
            except Exception as e:
                logger.warning(f"Error en búsqueda de jurisprudencia para query '{query}': {e}")
        
        return {
            'jurisprudencia': jurisprudencia_encontrada,
            'total_encontrados': len(jurisprudencia_encontrada),
            'queries_utilizadas': queries
        }
        
        
    def buscar_normativa_multindice(self, query: str, k: int = 3) -> List[Dict]:
        """Busca en los tres índices legislativos y devuelve los k mejores resultados globales"""
        hits = []
        for nombre, client in self.norma_clients.items():
            try:
                results = client.search(search_text=query, top=k)
                for r in results:
                    hits.append({
                        "articulo": r.get("articulo", ""),
                        "texto": r.get("texto", "")[:600],      # recortamos
                        "score": r.get("@search.score", 0)
                    })
            except Exception as e:
                logger.warning(f"Búsqueda falló en {nombre}: {e}")
                    
        # ordenar por score descendente y quedarnos con los k mejores globales
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:k]
        print(hits)
        return hits
    
    def _buscar_normativa_relevante(self, normas: List[str], requisitos: List[str]) -> List[Dict]:
        """Busca artículos normativos relevantes desde los índices vectoriales usando Azure AI Search"""
        index_names = ["resolucion_normativa", "decreto_reglamentario", "codigo_tributario"]
        normativa_encontrada = []
        
        for index_name in index_names:
            client = SearchClient(
            endpoint="https://search-caba-poc-eastus-001.search.windows.net",
            index_name=index_name,
            credential=AzureKeyCredential(os.environ.get("AZURE_SEARCH_KEY_RESOLUCION"))
            )
            
            for criterio in normas + requisitos:
                results = client.search(
                search_text=criterio,
                top=2,
                search_fields=["texto", "articulo"]
            )
                for result in results:
                    normativa_encontrada.append({
                    "articulo": result.get("articulo", ""),
                    "texto": result.get("texto", ""),
                    "score": result.get("@search.score", 0),
                    "fuente": index_name
                })
        return normativa_encontrada
        
   


    def validar_cumplimiento_avanzado(self, 
                                     datos_corrida: DatosCorridaVista,
                                     clasificacion: Dict,
                                     hechos: Dict,
                                     jurisprudencia: Dict,
                                     normativa_relevante: List[Dict]) -> ResultadoValidacion:
        """Validación integral con análisis jurisprudencial"""
        norma_chunks = self.buscar_normativa_multindice(
            query=" ".join(datos_corrida.normativa_aplicada + 
            [c['subtipo'] for c in clasificacion.get('clasificaciones', [])[:1]])
        )
        logger.info(f"Normativa añadida al prompt: {len(norma_chunks)} chunks")
        normativa_json = json.dumps(norma_chunks, indent=2, ensure_ascii=False)
        
        with open("service/prompt/automization/prompt_validar_cumplimiento.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        
        prompt = prompt_template.format(
                datos_corrida_contribuyente=datos_corrida.contribuyente,
                datos_corrida_cuit=datos_corrida.cuit,
                datos_corrida_tributo=datos_corrida.tributo,
                requisitos_cumplir=json.dumps(datos_corrida.requisitos_para_justificar, ensure_ascii=False),
                normativa_aplicada=json.dumps(datos_corrida.normativa_aplicada, ensure_ascii=False),
                impugnaciones_clasificadas=json.dumps(clasificacion, indent=2, ensure_ascii=False),
                hechos_extraidos=json.dumps(hechos, indent=2, ensure_ascii=False),
                jurisprudencia_relevante=json.dumps(jurisprudencia, indent=2, ensure_ascii=False),
                normativa_relevante_pqrs=normativa_relevante
            )
        
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3
            )
            raw_response_content = response.choices[0].message.content.strip()
            # Elimina las comillas triples y la palabra "json" si existen
            if raw_response_content.startswith("```json"):
                json_string = raw_response_content[len("```json"):].strip()
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].strip()
            else:
                json_string = raw_response_content
            resultado = json.loads(json_string)
            
            return ResultadoValidacion(
                cumplimiento_general=resultado.get('cumplimiento_general', 'no_cumple'),
                detalles_por_requisito=resultado.get('evaluacion_por_requisito', []),
                recomendacion_resolucion=resultado.get('recomendacion_resolucion', ''),
                fundamentos_juridicos=resultado.get('fundamentos_principales', [])
            )
            
        except Exception as e:
            logger.error(f"Error en validación avanzada: {e}")
            raise

    def generar_resolucion_estructurada(self, 
                                       datos_corrida: DatosCorridaVista,
                                       validacion: ResultadoValidacion,
                                       tipo_resolucion: str = "determinativa") -> str:
        """Genera resolución siguiendo plantillas oficiales"""
        
        plantillas = {
            "determinativa": {
                "estructura": ["VISTOS", "RESULTANDO", "CONSIDERANDO", "RESUELVE"],
                "formato": "Resolución Determinativa N° XXX/2025"
            },
            "reconsideracion": {
                "estructura": ["VISTOS", "RESULTANDO", "CONSIDERANDO", "RESUELVE"],
                "formato": "Resolución de Reconsideración N° XXX/2025"
            }
        }
        
        plantilla = plantillas.get(tipo_resolucion, plantillas["determinativa"])
        
        prompt = f"""
        Redacta una {plantilla['formato']} siguiendo estrictamente el formato oficial de la DGR Córdoba.
        
        DATOS DEL EXPEDIENTE:
        {json.dumps(datos_corrida.__dict__, indent=2, ensure_ascii=False)}
        
        RESULTADO DE LA EVALUACIÓN:
        {json.dumps(validacion.__dict__, indent=2, ensure_ascii=False)}
        
        ESTRUCTURA OBLIGATORIA:
        {plantilla['estructura']}
        
        INSTRUCCIONES ESPECÍFICAS:
        1. VISTOS: Citar expediente, corrida de vista, descargo presentado
        2. RESULTANDO: Narrar cronológicamente las actuaciones
        3. CONSIDERANDO: Análisis jurídico fundado en normativa y jurisprudencia
        4. RESUELVE: Artículos dispositivos claros y precisos
        
        ESTILO: Formal, técnico-jurídico, sin ambigüedades
        EXTENSIÓN: 4-6 páginas aproximadamente
        
        La resolución debe ser completa y lista para firma.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1,
                max_tokens=4000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error al generar resolución: {e}")
            raise

    def procesar_caso_completo(self, 
                              texto_corrida: str,
                              textos_descargo: List[str],
                              tipo_resolucion: str = "determinativa") -> Dict:
        """Procesamiento completo de un caso de DO"""
        
        logger.info("Iniciando procesamiento completo del caso")
        
        try:
            # 1. Extraer datos de corrida de vista
            datos_corrida = self.extraer_datos_corrida_vista(texto_corrida)
            logger.info(f"Datos extraídos para {datos_corrida.contribuyente}")
            
            # 2. Clasificar impugnaciones
            clasificacion = self.clasificar_impugnacion_avanzada(textos_descargo)
            logger.info(f"Impugnaciones clasificadas: {len(clasificacion.get('clasificaciones', []))}")
            
            # 3. Extraer hechos
            hechos = self.extraer_hechos_estructurados(textos_descargo, texto_corrida)
            logger.info(f"Hechos extraídos: {len(hechos.get('hechos', []))}")
            
            # 4. Buscar jurisprudencia
            jurisprudencia = self.buscar_jurisprudencia_contextual(clasificacion, hechos)
            logger.info(f"Jurisprudencia encontrada: {jurisprudencia['total_encontrados']} resultados")
            
            normativa_relevante = self._buscar_normativa_relevante(
                normas=datos_corrida.normativa_aplicada,
                requisitos=datos_corrida.requisitos_para_justificar
            )

            # 5. Validar cumplimiento
            validacion = self.validar_cumplimiento_avanzado(
                datos_corrida, clasificacion, hechos, jurisprudencia, normativa_relevante
            )
            logger.info(f"Validación completada: {validacion.cumplimiento_general}")
            
            # 6. Generar resolución
            resolucion = self.generar_resolucion_estructurada(
                datos_corrida, validacion, tipo_resolucion
            )
            logger.info("Resolución generada exitosamente")
            
            return {
                "datos_corrida": datos_corrida.__dict__,
                "clasificacion": clasificacion,
                "hechos": hechos,
                "jurisprudencia": jurisprudencia,
                "validacion": validacion.__dict__,
                "resolucion": resolucion,
                "timestamp": datetime.now().isoformat(),
                "estado": "completado"
            }
            
        except Exception as e:
            logger.error(f"Error en procesamiento completo: {e}")
            return {
                "estado": "error",
                "mensaje": str(e),
                "timestamp": datetime.now().isoformat()
            }



# Grafo de procesamiento en LangGraph

class EstadoCaso(TypedDict, total=False):
    texto_corrida: str
    textos_descargo: List[str]
    datos_corrida: DatosCorridaVista
    clasificacion: dict
    hechos: dict
    jurisprudencia: dict
    normativa_relevante: List[Dict]
    validacion: ResultadoValidacion
    resolucion: str
    timestamp: str
    estado: str


def construir_grafo() -> StateGraph:
    """Compila y devuelve el grafo de LangGraph para el flujo completo"""

    sistema = SistemaAutomatizacionDO()

    def extraer_corrida(state: EstadoCaso) -> EstadoCaso:
        datos = sistema.extraer_datos_corrida_vista(state["texto_corrida"])
        return {**state, "datos_corrida": datos}

    def clasificar_impugnacion(state: EstadoCaso) -> EstadoCaso:
        clasificacion = sistema.clasificar_impugnacion_avanzada(state["textos_descargo"])
        return {**state, "clasificacion": clasificacion}

    def extraer_hechos(state: EstadoCaso) -> EstadoCaso:
        # Pasar tanto la corrida como los descargos
        hechos = sistema.extraer_hechos_estructurados(
            state["textos_descargo"], 
            state["texto_corrida"]  # ← AGREGAR ESTO
        )
        return {**state, "hechos": hechos}

    def buscar_jurisprudencia(state: EstadoCaso) -> EstadoCaso:  # Cambié el nombre de la función
        juris = sistema.buscar_jurisprudencia_contextual(
            state["clasificacion"], state["hechos"]
        )
        return {**state, "jurisprudencia": juris}

    def buscar_normativa(state: EstadoCaso) -> EstadoCaso:
        normativa = sistema._buscar_normativa_relevante(
            normas=state["datos_corrida"].normativa_aplicada,
            requisitos=state["datos_corrida"].requisitos_para_justificar,
        )
        return {**state, "normativa_relevante": normativa}

    def validar(state: EstadoCaso) -> EstadoCaso:
        validacion = sistema.validar_cumplimiento_avanzado(
            state["datos_corrida"],
            state["clasificacion"],
            state["hechos"],
            state["jurisprudencia"],
            state["normativa_relevante"],
        )
        return {**state, "validacion": validacion}

    def resolver(state: EstadoCaso) -> EstadoCaso:
        resolucion = sistema.generar_resolucion_estructurada(
            state["datos_corrida"], state["validacion"], tipo_resolucion="determinativa"
        )
        return {
            **state,
            "resolucion": resolucion,
            "estado": "completado",
            "timestamp": datetime.now().isoformat(),
        }

    # ---- Construcción del grafo
    builder = StateGraph(EstadoCaso)

    builder.add_node("extraer_corrida", extraer_corrida)
    builder.add_node("clasificar_impugnacion", clasificar_impugnacion)
    builder.add_node("extraer_hechos", extraer_hechos)
    builder.add_node("buscar_jurisprudencia_node", buscar_jurisprudencia)  # Cambié el nombre del nodo
    builder.add_node("buscar_normativa", buscar_normativa)
    builder.add_node("validar", validar)
    builder.add_node("resolver", resolver)

    builder.set_entry_point("extraer_corrida")
    builder.add_edge("extraer_corrida", "clasificar_impugnacion")
    builder.add_edge("clasificar_impugnacion", "extraer_hechos")
    builder.add_edge("extraer_hechos", "buscar_jurisprudencia_node")  # Actualicé la referencia
    builder.add_edge("buscar_jurisprudencia_node", "buscar_normativa")  # Actualicé la referencia
    builder.add_edge("buscar_normativa", "validar")
    builder.add_edge("validar", "resolver")
    builder.add_edge("resolver", END)

    return builder.compile()



# Ejemplo de uso rápido

if __name__ == "__main__":
    # Compilar grafo
    grafo = construir_grafo()
    
    rutas_archivos = {
    "corrida": "C:/Users/Usuario/Downloads/POC-CBA/fiscalizacion_classification/data/casos/extract/Caso_2/corrida/caso2_corrida_de_vista-SD_extraido.txt",
    "aportes": [
        "C:/Users/Usuario/Downloads/POC-CBA/fiscalizacion_classification/data/casos/extract/Caso_2/aporte/caso2_aporte_1-SD_extraido.txt",
        "C:/Users/Usuario/Downloads/POC-CBA/fiscalizacion_classification/data/casos/extract/Caso_2/aporte/caso2_aporte_2-SD_extraido.txt",
        "C:/Users/Usuario/Downloads/POC-CBA/fiscalizacion_classification/data/casos/extract/Caso_2/aporte/caso2_aporte_3-SD_extraido.txt",
        "C:/Users/Usuario/Downloads/POC-CBA/fiscalizacion_classification/data/casos/extract/Caso_2/aporte/caso2_aporte_4-SD_extraido.txt",
        "C:/Users/Usuario/Downloads/POC-CBA/fiscalizacion_classification/data/casos/extract/Caso_2/aporte/caso2_aporte_5-SD_extraido.txt",
        "C:/Users/Usuario/Downloads/POC-CBA/fiscalizacion_classification/data/casos/extract/Caso_2/aporte/caso2_aporte_6-SD_extraido.txt"
    ]
}






    
    with open(rutas_archivos["corrida"], "r", encoding="utf-8") as f:
        contenido_corrida = f.read()
        
    contenidos_aportes = []
    for ruta in rutas_archivos["aportes"]:
        with open(ruta, "r", encoding="utf-8") as f:
            contenidos_aportes.append(f.read())

    # Estado inicial de prueba
    estado_inicial = {
        "texto_corrida": contenido_corrida,
        "textos_descargo": contenidos_aportes
    }

    resultado_final = grafo.invoke(estado_inicial)
    
    # Convertir objetos no serializables a diccionarios
    resultado_serializable = {}
    for key, value in resultado_final.items():
        if hasattr(value, '__dict__'):
            # Si el objeto tiene __dict__, convertirlo a diccionario
            resultado_serializable[key] = value.__dict__
        else:
            resultado_serializable[key] = value 
            
    with open("resultado_procesamiento.json", "w", encoding="utf-8") as f:
            json.dump(resultado_serializable, f, indent=2, ensure_ascii=False)
    
    print(json.dumps(resultado_serializable, indent=2, ensure_ascii=False))
