# visualizador_fiscalizacion.py

import streamlit as st
import json
import os

# CONFIGURACIÓN
st.set_page_config(
    page_title="Panel de Auditoría Fiscal AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FUNCIONES DE VISUALIZACIÓN
def seccion_datos_corrida(data):
    st.subheader("🧾 Datos del Contribuyente y Fiscalización")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Contribuyente:** {data['contribuyente']}")
        st.markdown(f"**CUIT:** {data['cuit']}")
        st.markdown(f"**Tributo:** {data['tributo']}")
        st.markdown(f"**Actividad:** {data['actividad_codificada']}")
    with col2:
        st.markdown(f"**Períodos Fiscales:** {', '.join(data['periodos_fiscales'])}")
        st.markdown(f"**Multa estimada:** {data['multa_estimativa']}")
        st.markdown(f"**Plazo para descargo:** {data['plazo_para_descargo']} días")
        st.markdown(f"**Canal de presentación:** {data['canal_presentacion']}")

    st.markdown("**Normativa Aplicada:**")
    st.code("\n".join(data['normativa_aplicada']), language="markdown")

    st.markdown("**Motivo del Ajuste:**")
    st.warning(data["motivo_del_ajuste"])


def seccion_hechos(hechos):
    st.subheader("📚 Hechos Relevantes Identificados")

    tipo_filtro = st.selectbox("Filtrar por tipo de hecho", options=["Todos"] + list(set(h['tipo'] for h in hechos)))
    relevancia_filtro = st.selectbox("Filtrar por relevancia jurídica", options=["Todos"] + list(set(h['relevancia_juridica'] for h in hechos)))

    for hecho in hechos:
        if (tipo_filtro != "Todos" and hecho['tipo'] != tipo_filtro) or \
           (relevancia_filtro != "Todos" and hecho['relevancia_juridica'] != relevancia_filtro):
            continue

        with st.expander(f"📌 {hecho['descripcion']}"):
            st.markdown(f"- **Tipo:** {hecho['tipo']}")
            st.markdown(f"- **Relevancia jurídica:** {hecho['relevancia_juridica']}")
            st.markdown(f"- **Requiere acreditación:** `{hecho['requiere_acreditacion']}`")
            st.markdown(f"- **Evidencia:** {hecho['evidencia_textual']}")
            st.markdown(f"- **Observaciones:** {hecho['observaciones']}")


def seccion_clasificacion(clasif):
    st.subheader("📊 Clasificación del Descargo")
    for clas in clasif:
        with st.expander(f"🧷 {clas['tipo']} - {clas['subtipo']}"):
            st.markdown(f"- **Evidencia:** {clas['evidencia_textual']}")
            st.markdown(f"- **Solidez jurídica:** {clas['solidez_juridica']}")
            st.markdown(f"- **Fundamento normativo:** {clas['fundamento_normativo']}")
            st.markdown(f"- **Observaciones:** {clas['observaciones']}")


def seccion_jurisprudencia(data):
    st.subheader("⚖️ Jurisprudencia y Normativa Relevante")
    for j in data.get("jurisprudencia", {}).get("jurisprudencia", []):
        with st.expander(f"📎 {j.get('tipo', 'Sin tipo')}"):
            st.markdown(f"- **Agravio:** {j.get('agravio')}")
            st.markdown(f"- **Respuesta:** {j.get('respuesta')}")

    st.markdown("### 📘 Normativa Relevante")
    for n in data.get("normativa_relevante", []):
        st.markdown(f"- **{n['articulo']}:** {n['texto']}")


def seccion_textos(texto_corrida, textos_descargo):
    st.subheader("🗂 Textos Extraídos del Caso")
    with st.expander("📄 Corrida de Vista"):
        st.text_area("Texto completo de la Corrida", value=texto_corrida, height=300)

    for i, txt in enumerate(textos_descargo):
        with st.expander(f"📥 Aporte del Contribuyente {i+1}"):
            st.text_area(f"Texto del aporte {i+1}", value=txt, height=300)


def seccion_recomendaciones(clasificacion):
    st.subheader("📌 Estrategia de Defensa y Recomendaciones")
    st.markdown("**Resumen Estratégico:**")
    st.info(clasificacion.get("resumen_estrategia_defensa", "No disponible"))

    st.markdown("**Puntos Débiles Identificados:**")
    for debil in clasificacion.get("puntos_debiles_impugnacion", []):
        st.warning(f"- {debil}")

    st.markdown("**Recomendación General:**")
    st.success(clasificacion.get("recomendacion_respuesta", "No disponible"))


# INICIO DEL VISUALIZADOR
st.title("🔍 Panel Inteligente de Auditoría Fiscal (LLM)")

# Cargar automáticamente desde ejecución LLM
json_path = "output_resultado_llm.json"
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        datos = json.load(f)
else:
    datos = None

# UI de navegación
if datos:
    menu = st.sidebar.radio("📌 Secciones disponibles", [
        "Datos del Caso",
        "Hechos Relevantes",
        "Clasificación del Descargo",
        "Jurisprudencia y Normativa",
        "Estrategia y Recomendaciones",
        "Textos Extraídos"
    ])

    if menu == "Datos del Caso":
        seccion_datos_corrida(datos["datos_corrida"])
    elif menu == "Hechos Relevantes":
        seccion_hechos(datos["hechos"]["hechos"])
    elif menu == "Clasificación del Descargo":
        seccion_clasificacion(datos["clasificacion"]["clasificaciones"])
    elif menu == "Jurisprudencia y Normativa":
        seccion_jurisprudencia(datos)
    elif menu == "Estrategia y Recomendaciones":
        seccion_recomendaciones(datos["clasificacion"])
    elif menu == "Textos Extraídos":
        seccion_textos(datos["texto_corrida"], datos["textos_descargo"])
else:
    st.info("🔄 Aún no se ha generado el archivo de salida JSON desde el modelo.")
    st.stop()