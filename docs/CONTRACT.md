# Contrato entre los 3 repositorios de docu-sentinel

> Este archivo es **idéntico en los tres repos** (`docu-sentinel-frontend`, `docu-sentinel-backend`,
> `docu-sentinel-ai`). Si cambias algo aquí, cámbialo en los tres.
> Código, identificadores y nombres de campo: inglés. Mensajes para el usuario: español.

## 1. Mapa del sistema

```
┌────────────────────────────┐   REST /api/v1 + socket.io    ┌─────────────────────────┐   REST /v1    ┌──────────────────────┐
│ docu-sentinel-frontend     │ ────────────────────────────► │ docu-sentinel-backend   │ ────────────► │ docu-sentinel-ai     │
│ Next.js 14 · :3000         │ ◄──────────────────────────── │ NestJS · :3001          │ ◄──────────── │ FastAPI · :8000      │
│ Motor + Web Workers (local)│                               │ historial, líneas base  │               │ Isolation Forest     │
└────────────────────────────┘                               └─────────────────────────┘               └──────────────────────┘
```

* **El análisis de archivos ocurre en el navegador** (Web Workers). El archivo **no se sube** a ningún servidor.
* El backend solo recibe **resultados** (hallazgos) y **matrices de características numéricas** (sin textos originales)
  cuando el usuario decide guardar el historial o entrenar una línea base.
* La IA del servidor (línea base histórica) es **opcional**: si no está disponible, el frontend usa su
  Isolation Forest local.

Puertos por defecto: frontend `3000`, backend `3001`, ai `8000`.

## 2. Convención de características (features)

La construye el frontend en sus workers y la consume la IA. El **orden de columnas importa** y debe coincidir
entre entrenamiento y puntuación.

| Nombre de columna    | `kind`    | Significado                                             |
|----------------------|-----------|---------------------------------------------------------|
| `<field>`            | `numeric` | valor numérico del campo                                |
| `<field>`            | `date`    | fecha como **días desde 1970-01-01** (epoch days)       |
| `<field>__missing`   | `missing` | 1 si el campo está vacío, 0 si no                       |
| `<field>__rarity`    | `rarity`  | `-ln(frecuencia relativa de la categoría)` (más alto = más raro) |

Valores desconocidos/no numéricos se envían como `null` (JSON no admite `NaN`).
`schemaSignature` (opcional) = hash estable de `columns.join('|')` para comprobar compatibilidad rápido.

**Modelo (idéntico en el frontend y en la IA):** escalado robusto (mediana/IQR) → escala **log** para columnas
numéricas positivas y asimétricas (dinero, cantidades) → Isolation Forest + guarda de extrapolación + modelo de
**coherencia** (Mahalanobis recortado, detecta relaciones rotas entre campos). Puntaje final = máximo de los tres,
en (0,1). Una fila se marca si `score ≥ max(threshold_floor, cuantil 1-contamination del entrenamiento)`.

| Sensibilidad (frontend) | `contamination` | `threshold_floor` |
|-------------------------|-----------------|-------------------|
| `low`                   | 0.005           | 0.65              |
| `medium`                | 0.01            | 0.60              |
| `high`                  | 0.03            | 0.55              |

## 3. Servicio IA (`docu-sentinel-ai`) — `http://localhost:8000`

Cabecera opcional `X-API-Key` (si el servicio define `AI_API_KEY`). Toda respuesta trae `X-Request-ID`
(se reutiliza el recibido si es seguro: `[A-Za-z0-9._-]{1,64}`). Cuerpo máximo `MAX_BODY_MB` (64 MB por defecto).

| Método | Ruta                         | Descripción |
|--------|------------------------------|-------------|
| GET    | `/health`                    | `{status:"ok", version, engine, models, store: "memory"\|"disk"\|"postgres"}` |
| GET    | `/metrics`                   | Prometheus (texto). `Authorization: Bearer <METRICS_TOKEN>` si está definido |
| POST   | `/v1/models`                 | Entrena y guarda un modelo (línea base) → `ModelInfo` (201) |
| GET    | `/v1/models`                 | Lista `ModelInfo[]` |
| GET    | `/v1/models/{id}`            | `ModelInfo` |
| DELETE | `/v1/models/{id}`            | 204 |
| POST   | `/v1/models/{id}/score`      | Puntúa filas nuevas contra la línea base → `ScoreResponse` |
| POST   | `/v1/detect`                 | Sin estado: entrena y puntúa el mismo lote → `DetectResponse` |

**TrainRequest / DetectRequest**
```json
{ "name": "Facturas 2025", "columns": ["quantity","unit_price","city__rarity"],
  "kinds": ["numeric","numeric","rarity"],            // opcional (se infiere por sufijo)
  "matrix": [[3, 120.5, 0.4], [2, null, 0.7]],        // máx. 200 000 filas / 8 000 000 celdas
  "contamination": 0.01, "n_estimators": 200, "seed": 42, "schema_signature": "ab12…",
  "threshold_floor": 0.6 }                            // opcional, 0.5 ≤ x < 1, por defecto 0.55
```
`DetectRequest` añade `explain` y `top_k`.

**ModelInfo**: `{ model_id, name, columns, kinds, rows, threshold, threshold_floor, contamination, seed, trained_at,
schema_signature, engine }` (`engine = "sklearn-isolation-forest+coherence"`).

**ScoreRequest**: `{ columns, kinds?, matrix, explain: "flagged"|"all"|"none" = "flagged", top_k: 1..10 = 3, threshold?: 0..1 }`

**ScoreResponse**
```json
{ "model_id": "mdl_ab12cd34ef56", "threshold": 0.6123,
  "scores": [0.51, 0.93], "flags": [false, true],
  "contributions": { "1": [ { "column":"total","field":"total","kind":"numeric","value":250000,
      "median":1200,"deviation":41.3,
      "message":"El valor 250.000 en «total» está 41,3 desviaciones robustas por encima de la mediana (1.200)." } ] } }
```
`scores` ∈ (0,1), mayor = más anómalo. `contributions` solo trae las filas pedidas por `explain`
(clave = índice de fila como texto). `DetectResponse` = `ScoreResponse` + `rows` + `engine`.

`Contribution.kind` ∈ `numeric | date | missing | rarity | coherence`. Con `coherence` el valor del campo no encaja
con lo que predicen los demás campos del registro; `deviation` es el residuo condicional y `expected` el valor
esperado (solo para `coherence`):
```json
{ "column":"total","field":"total","kind":"coherence","value":2261,"median":1255.1,"deviation":37.5,"expected":254.9,
  "message":"El valor 2.261 en «total» no es coherente con el resto del registro: según los demás campos se esperaría alrededor de 254,9." }
```
Las incoherencias solo se reportan cuando todos los valores de la fila están dentro del rango de entrenamiento; si un
valor es extremo por sí mismo, se explica ese valor (`numeric`/`date`/`rarity`).

**Errores**: `{ "code": "...", "message": "<español>" }` — códigos: `MODEL_NOT_FOUND` (404), `SCHEMA_MISMATCH` (422),
`INVALID_TRAINING_DATA` (422), `MODEL_LIMIT` (409), `UNAUTHORIZED` (401), `PAYLOAD_TOO_LARGE` (413),
`RATE_LIMITED` (429, con `Retry-After`; solo si `RATE_LIMIT_PER_MIN` > 0).

Persistencia de modelos: `DATABASE_URL` → tabla `ds_ai_models` en PostgreSQL; si no, `MODELS_DIR` (disco) o memoria.
`MODEL_TTL_DAYS` borra modelos viejos (al arrancar y antes de cada entrenamiento). Los errores de validación de forma
(`422`) siguen el formato estándar de FastAPI (`detail[]`).

## 4. Backend (`docu-sentinel-backend`) — `http://localhost:3001/api/v1`

Swagger en `http://localhost:3001/docs`. Cabecera opcional `X-API-Key` para escrituras (si `API_KEY` está definido).
Errores: `{ "statusCode": 400, "code": "VALIDATION_ERROR", "message": "..." }`. Toda respuesta trae `X-Request-ID`
(expuesta por CORS). Retención opcional: con `RETENTION_DAYS=N` el backend borra los reportes con más de N días
(emite `analysis.deleted` por cada uno).

### Tipos compartidos
```ts
type RiskLevel = 'low' | 'medium' | 'high' | 'critical';
type FileFormat = 'csv' | 'tsv' | 'json' | 'ndjson' | 'pdf' | 'xlsx';
type ProcessingMode = 'main' | 'worker' | 'pool';
type AnomalyType =
  | 'duplicate_exact' | 'duplicate_key' | 'duplicate_near' | 'missing_field' | 'empty_record'
  | 'type_mismatch' | 'invalid_format' | 'impossible_date' | 'inconsistent_label' | 'whitespace'
  | 'out_of_range' | 'negative_value' | 'future_date' | 'impossible_relation'
  | 'suspicious_threshold' | 'suspicious_round' | 'benford' | 'multivariate_anomaly' | 'mixed_types';

interface FindingDto {
  recordIndex: number | null;   // 0-based global row index; null = column-level finding
  recordId?: string;            // value of the detected key column, if any
  type: AnomalyType;
  risk: RiskLevel;
  score: number;                // 0..100
  field: string | null;
  value?: string;               // preview, <= 120 chars
  explanation: string;          // Spanish
  detector: string;             // e.g. "range", "isolation_forest"
}
interface ColumnSummaryDto {
  name: string;
  type: 'integer' | 'decimal' | 'date' | 'boolean' | 'email' | 'string' | 'mixed';
  nullRate: number;             // 0..1
  distinct: number | null;
}
```

### Cuentas de usuario (solo si el backend define `JWT_SECRET`)
Sin `JWT_SECRET` todo funciona como antes (sin login). Con `JWT_SECRET`, **todas** las rutas de `/analyses` y
`/baselines` exigen `Authorization: Bearer <token>` (si falta o expiró: `401 AUTH_REQUIRED`) y cada reporte y línea base
pertenece a quien lo creó: los de otro usuario responden `404`.

| Método | Ruta | Cuerpo | Respuesta |
|--------|------|--------|-----------|
| POST | `/auth/register` | `{ email, password (8–200), name? }` | `AuthSession` (201) · `409 EMAIL_TAKEN` · `403 REGISTRATION_DISABLED` |
| POST | `/auth/login` | `{ email, password }` | `AuthSession` · `401 INVALID_CREDENTIALS` |
| GET | `/auth/me` | | `User` |

```ts
interface User { id: string; email: string; name: string; createdAt: string }
interface AuthSession { token: string; expiresAt: string; user: User }   // token = JWT HS256, JWT_TTL_HOURS (168)
```
El frontend guarda el token en `localStorage` y en la cookie `ds_token` (para que el SSR de `/reportes` lo reenvíe).

### Análisis (historial de reportes)
| Método | Ruta | Cuerpo / query | Respuesta |
|--------|------|----------------|-----------|
| POST | `/analyses` | `SaveAnalysisDto` | `AnalysisSummary` (201) |
| GET | `/analyses` | `?limit=20&offset=0` | `{ items: AnalysisSummary[], total }` |
| GET | `/analyses/:id` | | `AnalysisDetail` (= summary + `columns`) |
| GET | `/analyses/:id/findings` | `?risk=high&type=out_of_range&field=total&q=texto&sort=score:desc&limit=100&offset=0` | `{ items: FindingDto[], total }` |
| GET | `/analyses/:id/export.csv` | | CSV UTF-8 con BOM, cabeceras: Registro, Tipo de anomalía, Nivel de riesgo, Campo afectado, Explicación |
| GET | `/analyses/:id/export.json` | | `{ analysis: AnalysisDetail, findings: FindingDto[] }` (todos, por puntaje) |
| DELETE | `/analyses/:id` | | 204 |

```ts
interface SaveAnalysisDto {
  fileName: string; format: FileFormat; fileSizeBytes: number; rows: number;
  columns: ColumnSummaryDto[]; durationMs: number; mode: ProcessingMode; workers: number;
  sensitivity: 'low' | 'medium' | 'high'; mlMode: 'off' | 'local' | 'baseline'; baselineId?: string;
  findings: FindingDto[];       // máx. 100 000 (el frontend recorta por score y lo avisa)
}
interface AnalysisSummary {
  id: string; createdAt: string; fileName: string; format: FileFormat; fileSizeBytes: number;
  rows: number; columnCount: number; durationMs: number; mode: ProcessingMode; workers: number;
  findingsCount: number; recordsAffected: number;
  byRisk: Record<RiskLevel, number>; byType: Record<string, number>;
}
```

### Líneas base históricas (el backend las delega a la IA)
| Método | Ruta | Cuerpo | Respuesta |
|--------|------|--------|-----------|
| POST | `/baselines` | `{ name, columns, kinds?, matrix, schemaSignature?, contamination?, seed?, thresholdFloor? }` | `BaselineDto` (201) |
| GET | `/baselines` | | `BaselineDto[]` |
| GET | `/baselines/:id` | | `BaselineDto` |
| DELETE | `/baselines/:id` | | 204 (también borra el modelo en la IA, best-effort) |
| POST | `/baselines/:id/retrain` | `{ matrix, columns?, contamination?, thresholdFloor?, seed? }` (mismas columnas y orden) | `BaselineDto` con `version` + 1 (el modelo anterior se borra) · `422 SCHEMA_MISMATCH` si cambian las columnas |
| POST | `/baselines/:id/score` | `{ matrix, explain?, topK?, schemaSignature? }` (las columnas las pone el backend) | `ScoreResponse` de la IA |

```ts
interface BaselineDto {
  id: string; name: string; aiModelId: string; columns: string[]; kinds: string[];
  rows: number; threshold: number; trainedAt: string; schemaSignature?: string;
  thresholdFloor?: number;      // 0.5..0.99, el que usó la IA al entrenar
  version?: number;             // 1 al crear, +1 en cada reentrenamiento
}
```
`thresholdFloor` (0.5–0.99) se reenvía a la IA como `threshold_floor`. Si en `/score` se envía `schemaSignature`, la línea
base tiene firma y no coinciden → `422 { code: "SCHEMA_MISMATCH" }` **sin** llamar a la IA.
Si la IA no responde: `503 { code: "AI_UNAVAILABLE", message: "El servicio de IA no está disponible…" }`
→ el frontend cae al Isolation Forest local.

### Tiempo real (socket.io)
Namespace `/realtime` (ruta por defecto `/socket.io`). Eventos servidor→cliente:
`analysis.saved` (`AnalysisSummary`), `analysis.deleted` (`{id}`), `baseline.trained` (`BaselineDto`, también al
reentrenar), `baseline.deleted` (`{id}`). Con cuentas activas el cliente se conecta con `io(url, { auth: { token } })`
y solo recibe **sus** eventos; sin token recibe `auth.error` y se desconecta.

### Salud
`GET /api/v1/health` → `{ status: "ok", ai: "up" | "down", version, auth: boolean, store: "postgres" | "files" }`.
`GET /metrics` (fuera del prefijo `/api/v1`) → Prometheus; `Authorization: Bearer <METRICS_TOKEN>` si está definido.

### Webhook opcional
Con `WEBHOOK_URL`, al guardar un reporte con hallazgos de riesgo ≥ `WEBHOOK_MIN_RISK` (`critical` por defecto) el
backend hace `POST` con `{ event: "analysis.flagged", minRisk, analysis: { id, createdAt, fileName, rows, findingsCount,
recordsAffected, byRisk } }`. Con `WEBHOOK_SECRET` añade `X-DocuSentinel-Signature: sha256=<HMAC-SHA256 del cuerpo>`.

### Almacenamiento
`DATABASE_URL` → PostgreSQL (`ds_analyses`, `ds_findings` con índices por riesgo/tipo/registro, `ds_kv` para líneas base
y usuarios; el esquema se crea solo). Sin `DATABASE_URL` → archivos JSON en `DATA_DIR`. Mismo comportamiento (las dos
implementaciones pasan las mismas pruebas de contrato).
