# Despliegue en Render

Este servicio se despliega junto con el backend y PostgreSQL desde el Blueprint del repo **docu-sentinel-backend**
([`render.yaml`](https://github.com/JuanMaya8/docu-sentinel-backend/blob/main/render.yaml) y
[`DEPLOY.md`](https://github.com/JuanMaya8/docu-sentinel-backend/blob/main/DEPLOY.md)): el Blueprint apunta a este repo
con `repo: https://github.com/JuanMaya8/docu-sentinel-ai`.

Si prefieres crearlo a mano: Render → **New → Web Service** → este repo → runtime **Docker**, *Health Check Path*
`/health`, y estas variables:

| Variable | Valor | Para qué |
|---|---|---|
| `DATABASE_URL` | *Internal Database URL* de tu PostgreSQL | los modelos sobreviven a reinicios (el disco gratuito se borra) |
| `AI_API_KEY` | el mismo valor que en el backend | solo el backend puede llamar a `/v1/*` |
| `MODEL_TTL_DAYS` | `180` | borra modelos viejos |
| `TRUST_PROXY` | `1` | IP real del cliente detrás del proxy de Render |
| `RATE_LIMIT_PER_MIN` | `0` (o p. ej. `600`) | límite por cliente en `/v1/*` |
| `METRICS_TOKEN` | opcional | protege `GET /metrics` |

`PORT` lo inyecta Render y el `Dockerfile` lo respeta. La tabla `ds_ai_models` se crea sola al arrancar.
