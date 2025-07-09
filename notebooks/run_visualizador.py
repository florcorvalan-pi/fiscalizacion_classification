import subprocess
import json
import os

print("🔄 Ejecutando modelo LLM desde fiscalizaciones_multiagente.py...")

base_dir = "/Users/rayespinoza/AI_Projects/fiscalizacion_classification"
multiagente_path = os.path.join(base_dir, "notebooks/fiscalizaciones_multiagente.py")
visualizador_path = os.path.join(base_dir, "notebooks/visualizador.py")
json_path = os.path.join(base_dir, "output_resultado_llm.json")

if not os.path.exists(multiagente_path):
    raise FileNotFoundError("No se encontró el archivo fiscalizaciones_multiagente.py")

# --- Ejecutar y mostrar en tiempo real ---
with subprocess.Popen(["python", multiagente_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
    for line in proc.stdout:
        print(line, end="")  # ya incluye salto de línea
    proc.wait()

# Verifica éxito
if proc.returncode != 0:
    raise RuntimeError("❌ Error durante la ejecución de fiscalizaciones_multiagente.py")

# Validar salida generada
if not os.path.exists(json_path):
    raise FileNotFoundError("No se generó output_resultado_llm.json")

print("✅ Archivo generado:", json_path)

# Lanzar visualizador
print("🚀 Lanzando visualizador en navegador...")
subprocess.Popen(["streamlit", "run", visualizador_path])
