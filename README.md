# docu-sentinel-ai

Servicio de **detección de anomalías** del proyecto *Motor Inteligente para Detectar Inconsistencias en Documentos*.
Es el tercer repositorio del sistema (junto a `docu-sentinel-frontend` y `docu-sentinel-backend`).

Recibe matrices numéricas de características, entrena un **Isolation Forest** (scikit-learn) y devuelve, por registro,
una puntuación de anomalía 0–1 con la **explicación en español** de qué campos lo hacen raro.
Permite **aprender de datos históricos**: se entrena una *línea base* con un archivo histórico y luego se puntúan
archivos nuevos contra ella.

> **Estado: avance funcional (v0.1.0)**, no el proyecto terminado. Lee la sección *Qué está hecho y qué no*.

## Cómo ejecutarlo

Requisitos: **Python 3.11+** (probado con 3.13).

```bash
# 1. Entorno virtual
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Dependencias (incluye pytest y httpx para las pruebas)
pip install -r requirements-dev.txt

# 3. Configuración (opcional)
cp .env.example .env                 # MODELS_DIR=./models persiste los modelos entre reinicios

# 4. Levantar el servicio
uvicorn app.main:app --reload --port 8000
#   Swagger interactivo:  http://localhost:8000/docs
#   Salud:                http://localhost:8000/health

# 5. Pruebas
python -m pytest -q                  # o: python -m unittest discover -s tests -t .
```

Con Docker:
```bash
docker build -t docu-sentinel-ai .
docker run -p 8000:8000 -v "$PWD/models:/data/models" docu-sentinel-ai
```

Prueba rápida con `curl` (modo sin estado: entrena y puntúa el mismo lote; necesita ≥ 20 filas):
```bash
python - <<'PY' > /tmp/body.json
import json, random
random.seed(1)
rows = [[random.randint(1, 20), random.gauss(100, 10)] for _ in range(300)]
rows[0][1] = 9000.0                      # un precio absurdo
print(json.dumps({"columns": ["quantity", "unit_price"], "matrix": rows}))
PY
curl -s -X POST localhost:8000/v1/detect -H 'content-type: application/json' -d @/tmp/body.json | head -c 600
```

## Arquitectura

```
app/
├── main.py        FastAPI: solo cablea HTTP ↔ servicio (CORS, API key, manejo de errores)
├── service.py     AnomalyService: casos de uso (train, score, detect, list, get, delete) — sin framework
├── engine.py      Motor: escalado robusto + Isolation Forest + guarda de extrapolación + contribuciones
├── explain_es.py  Plantillas de explicación en español
├── formatting.py  Formato es-CO (1.234,5) y fechas
├── store.py       ModelStore: registro en memoria + persistencia opcional (joblib)
├── schemas.py     Contrato pydantic v2 (entrada/salida)
└── errors.py      ServiceError(code, mensaje_es, status)
```

Capas: **HTTP (main.py) → servicio (service.py) → motor (engine.py) → almacén (store.py)**. Solo `main.py` importa
FastAPI; el resto se prueba sin servidor web.

### Cómo detecta anomalías
1. **Escalado robusto** por columna: mediana y escala `IQR/1.349` (si el IQR es 0, desviación estándar; si no, 1).
   Los valores nulos se imputan con la mediana y se recortan a ±50 desviaciones.
2. **Isolation Forest** (200 árboles, submuestra de 256, semilla fija → resultados reproducibles).
   `score = -score_samples` ∈ (0,1); ≈0.5 es normal, → 1 es muy anómalo.
3. **Umbral**: `max(0.55, cuantil(1 - contamination) de los scores de entrenamiento)`. `contamination` por
   defecto 1 %. El umbral absoluto evita marcar siempre un 1 % de filas en datos limpios.
4. **Guarda de extrapolación**: un Isolation Forest puntúa *bajo* los valores muy por fuera del rango visto
   en el entrenamiento (siguen la misma rama que el extremo de entrenamiento). En modo línea base eso dejaría pasar
   un total de 250 000 cuando el histórico llega a 3 000. Se mide cuánto excede cada valor el rango de entrenamiento
   (en unidades robustas) y se mapea a `[0.5, 1)`; el score final es `max(IF, guarda)`.
5. **Explicación**: para cada fila marcada se calculan las desviaciones robustas por característica y se devuelven
   las `top_k` más fuertes, con mensaje en español, p. ej.
   *«El valor 250.000 en «total» está 41,3 desviaciones robustas por encima de la mediana (1.200).»*
   Para `__missing` y `__rarity` hay plantillas propias.

El contrato completo con los otros dos repos está en [`docs/CONTRACT.md`](docs/CONTRACT.md).

### Patrón de renderizado
No aplica a este repo (es una API JSON). El patrón de renderizado del sistema (CSR + SSG + SSR por ruta) está
justificado en el README de `docu-sentinel-frontend`.

## Qué está hecho y qué no

| Estado | Elemento |
|--------|----------|
| ✅ Verificado (21 pruebas ejecutadas + 4 de API omitidas por falta de fastapi; Python 3.13, scikit-learn 1.9.1, numpy 2.5.3, pydantic 2.13.5) | Motor, servicio, almacén con persistencia, explicaciones, formato es-CO, validación de esquema |
| ⚠️ Escrito pero **no ejecutado** en el entorno donde se creó (no había `fastapi` instalable) | `app/main.py` (rutas FastAPI), `tests/test_api.py` (se omiten solos si falta fastapi), `Dockerfile` |
| ❌ No hecho | Autenticación real por usuario, modelos alternativos (autoencoder, LOF), reentrenamiento incremental, tablas de frecuencia de categorías dentro de la línea base, métricas de calidad sobre datos reales, observabilidad |

Tras `pip install -r requirements-dev.txt`, `python -m pytest -q` ejecuta además `tests/test_api.py`; si algo falla ahí,
es lo primero que hay que corregir.

## Limitaciones conocidas
* Las pruebas usan datos **sintéticos**. No hay validación con datos reales de una empresa.
* `joblib` usa *pickle*: solo carga modelos de un directorio propio y no accesible a terceros.
* Un modelo entrenado con pocas filas (< 200) es poco estable; el servicio exige ≥ 20 y recomienda ≥ 1 000.
* `rarity` en modo línea base se calcula en el frontend con las frecuencias del archivo nuevo, no con las del histórico.

## Despliegue (más adelante)
Imagen Docker lista (`Dockerfile`). Variables: `MODELS_DIR` (volumen persistente), `AI_API_KEY` (recomendado en
producción; el backend lo envía como `X-API-Key`), `CORS_ORIGINS` (normalmente vacío: solo el backend llama a este servicio).
Opciones simples: Render, Railway, Fly.io o Cloud Run (monta un volumen o usa un bucket si quieres que los modelos
sobrevivan a los reinicios).

## Nombre de repositorio sugerido
`docu-sentinel-ai`
