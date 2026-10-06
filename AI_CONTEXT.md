# AI_CONTEXT.md — contexto para la IA generadora de código

> Léeme completo antes de tocar nada. Este repo es **1 de 3** del proyecto *docu-sentinel*
> (Motor Inteligente para Detectar Inconsistencias en Documentos). Idioma: **código, nombres de archivo, commits y
> pruebas en inglés; textos para el usuario (mensajes de error, explicaciones) en español**.

## 1. Objetivo del proyecto (caso de estudio original)
Plataforma web donde el usuario carga CSV, JSON, PDF u otros documentos estructurados y el sistema busca: duplicados,
campos faltantes, inconsistencias, valores fuera de rango, relaciones imposibles, registros sospechosos y anomalías.
El procesamiento pesado se hace con **Web Workers** (parsing/transformación). El reporte muestra por hallazgo:
Registro · Tipo de anomalía · Nivel de riesgo · Campo afectado · Explicación. La **IA** es un modelo de detección de
anomalías que puede **aprender de datos históricos**. Diferencial: IA + archivos grandes + parsing + Web Workers +
análisis de anomalías, demostrando la diferencia de rendimiento **Main Thread vs. concurrente**.

## 2. Los 3 repos y el rol de ESTE repo
| Repo | Stack | Rol |
|------|-------|-----|
| `docu-sentinel-frontend` | Next.js 14 + TS | Motor de análisis local con Web Workers, UI, benchmark |
| `docu-sentinel-backend` | NestJS + TS | Historial de reportes, líneas base, proxy a la IA, WebSocket |
| **`docu-sentinel-ai`** (este) | FastAPI + scikit-learn | Isolation Forest, línea base histórica, explicaciones en español |

El contrato exacto entre los tres está en `docs/CONTRACT.md` (idéntico en los 3 repos). **No lo rompas sin cambiarlo
en los otros dos.**

## 3. Arquitectura de este repo
`main.py` (FastAPI, solo cableado) → `service.py` (casos de uso) → `engine.py` (matemática) → `store.py` (modelos).
`schemas.py` es el contrato pydantic. Solo `main.py` importa FastAPI.

Invariantes que NO se deben romper:
* `engine.py` y `service.py` **no importan FastAPI** (así se prueban sin servidor).
* Resultados **deterministas** con la misma `seed`.
* `ABS_THRESHOLD_FLOOR`, guarda de extrapolación y escalado robusto son decisiones medidas por tests
  (`test_baseline_mode_extrapolation_guard`); si los cambias, ajusta y vuelve a medir.
* Un `ModelStore` vacío es *falsy* por `__len__` → **nunca** uses `store or ModelStore()`; usa `is not None`
  (bug real ya corregido).
* Convención de columnas: `<field>`, `<field>__missing`, `<field>__rarity`; las plantillas de `explain_es.py` dependen de ella.
* Mensajes al usuario en español con formato es-CO (`fmt_number`).

## 4. Cómo ejecutar y probar
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                       # o: python -m unittest discover -s tests -t .
uvicorn app.main:app --reload --port 8000 # http://localhost:8000/docs
```

## 5. Estado exacto (hasta dónde llegué)
**Hecho y VERIFICADO** (ejecutado en el sandbox: Python 3.13, numpy 2.5.3, scikit-learn 1.9.1, pydantic 2.13.5,
comando `python3 -m unittest discover -s tests -t .` → 25 tests, OK, 4 omitidos):
* Escalado robusto, Isolation Forest, umbral, guarda de extrapolación, determinismo por semilla, manejo de `None`,
  columna constante, explicaciones en español (valor/missing/rarity), formato es-CO.
* Servicio: train/score/detect/list/get/delete, validación de esquema (`SCHEMA_MISMATCH`), errores con código,
  persistencia en disco con joblib y recarga tras "reinicio".

**Escrito pero NO ejecutado** (en el sandbox no se podía instalar `fastapi`: pip bloqueado):
* `app/main.py` (rutas, API key, CORS, manejador de `ServiceError`).
* `tests/test_api.py` (usa `fastapi.testclient`; se omite solo si falta fastapi).
* `Dockerfile`, `.github/workflows/ci.yml`.
→ **Primera tarea al retomar:** `pip install -r requirements-dev.txt && python -m pytest -q` y arreglar lo que falle en la capa HTTP.

**No hecho:**
* Auth por usuario / roles (solo API key compartida opcional).
* Almacenar en el modelo las **tablas de frecuencia de categorías** para que `__rarity` en modo línea base use las
  frecuencias históricas (hoy las calcula el frontend con el archivo nuevo).
* Otros algoritmos (LOF, autoencoder, ensembles), reentrenamiento incremental, versionado de modelos.
* Evaluación con datos reales (todas las pruebas son sintéticas).
* Métricas/observabilidad (Prometheus, logs estructurados), límites de tamaño a nivel de reverse proxy.
* Almacenamiento de modelos fuera del disco local (S3/GCS) para despliegues efímeros.

## 6. Checklist para llegar al 100 % (en orden)
1. Ejecutar y arreglar `tests/test_api.py` con fastapi real; ejecutar `docker build`.
2. Prueba de integración real con `docu-sentinel-backend` (entrenar línea base → puntuar) siguiendo `docs/CONTRACT.md`.
3. Guardar tablas de frecuencia en el modelo + endpoint para puntuar `rarity` con frecuencias históricas.
4. Evaluar con un dataset real (p. ej. facturas/inventario anonimizados): medir precisión/recobrado y recalibrar umbral.
5. Versionado de modelos y borrado por antigüedad; almacenamiento externo.
6. Observabilidad y límites (rate limit, tamaño de cuerpo, timeouts).
7. Desplegar (Render/Railway/Cloud Run) con `AI_API_KEY` y volumen para `MODELS_DIR`.

## 7. Prompts sugeridos para continuar
* "Lee AI_CONTEXT.md y docs/CONTRACT.md. Ejecuta `pip install -r requirements-dev.txt && python -m pytest -q`, corrige cualquier fallo de la capa FastAPI sin cambiar el contrato, y reporta."
* "Añade al modelo una tabla de frecuencias por columna `__rarity` y úsala en `score`; actualiza schemas, CONTRACT.md (en los 3 repos), pruebas y README."
* "Escribe una prueba de rendimiento: 200 000 filas × 40 columnas en `/v1/detect`; mide tiempo y memoria y propone límites."
