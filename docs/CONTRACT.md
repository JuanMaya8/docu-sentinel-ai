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

## 3. Servicio IA (`docu-sentinel-ai`) — `http://localhost:8000`

Cabecera opcional `X-API-Key` (si el servicio define `AI_API_KEY`).

| Método | Ruta                         | Descripción |
|--------|------------------------------|-------------|
| GET    | `/health`                    | `{status:"ok", version, engine, models}` |
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
  "contamination": 0.01, "n_estimators": 200, "seed": 42, "schema_signature": "ab12…" }
```
`DetectRequest` añade `explain` y `top_k`.

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

**Errores**: `{ "code": "...", "message": "<español>" }` — códigos: `MODEL_NOT_FOUND` (404), `SCHEMA_MISMATCH` (422),
`INVALID_TRAINING_DATA` (422), `MODEL_LIMIT` (409), `UNAUTHORIZED` (401). Los errores de validación de forma
(`422`) siguen el formato estándar de FastAPI (`detail[]`).

## 4. Backend (`docu-sentinel-backend`) — `http://localhost:3001/api/v1`

Swagger en `http://localhost:3001/docs`. Cabecera opcional `X-API-Key` para escrituras (si `API_KEY` está definido).
Errores: `{ "statusCode": 400, "code": "VALIDATION_ERROR", "message": "..." }`.

### Tipos compartidos
```ts
type RiskLevel = 'low' | 'medium' | 'high' | 'critical';
type FileFormat = 'csv' | 'tsv' | 'json' | 'ndjson' | 'pdf';
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

### Análisis (historial de reportes)
| Método | Ruta | Cuerpo / query | Respuesta |
|--------|------|----------------|-----------|
| POST | `/analyses` | `SaveAnalysisDto` | `AnalysisSummary` (201) |
| GET | `/analyses` | `?limit=20&offset=0` | `{ items: AnalysisSummary[], total }` |
| GET | `/analyses/:id` | | `AnalysisDetail` (= summary + `columns`) |
| GET | `/analyses/:id/findings` | `?risk=high&type=out_of_range&field=total&q=texto&sort=score:desc&limit=100&offset=0` | `{ items: FindingDto[], total }` |
| GET | `/analyses/:id/export.csv` | | CSV UTF-8 con BOM, cabeceras: Registro, Tipo de anomalía, Nivel de riesgo, Campo afectado, Explicación |
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
| POST | `/baselines` | `{ name, columns, kinds?, matrix, schemaSignature?, contamination?, seed? }` | `BaselineDto` (201) |
| GET | `/baselines` | | `BaselineDto[]` |
| GET | `/baselines/:id` | | `BaselineDto` |
| DELETE | `/baselines/:id` | | 204 (también borra el modelo en la IA, best-effort) |
| POST | `/baselines/:id/score` | `{ matrix, explain?, topK? }` (las columnas las pone el backend) | `ScoreResponse` de la IA |

```ts
interface BaselineDto {
  id: string; name: string; aiModelId: string; columns: string[]; kinds: string[];
  rows: number; threshold: number; trainedAt: string; schemaSignature?: string;
}
```
Si la IA no responde: `503 { code: "AI_UNAVAILABLE", message: "El servicio de IA no está disponible…" }`
→ el frontend cae al Isolation Forest local.

### Tiempo real (socket.io)
Namespace `/realtime` (ruta por defecto `/socket.io`). Eventos servidor→cliente:
`analysis.saved` (`AnalysisSummary`), `analysis.deleted` (`{id}`), `baseline.trained` (`BaselineDto`), `baseline.deleted` (`{id}`).

### Salud
`GET /api/v1/health` → `{ status: "ok", ai: "up" | "down", version }`.
