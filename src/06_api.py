# =============================================================================
# DESPLIEGUE EN AWS (escenario de produccion):
# 1. model_lgbm.pkl almacenado en s3://metro-medellin-datalake/models/
# 2. Esta API empaquetada como imagen Docker y desplegada en AWS ECS o Lambda
# 3. API Gateway expone el endpoint publico
# 4. En startup, la app descarga el pkl desde S3 con boto3
# Para correr localmente: uvicorn src.06_api:app --reload --port 8000
# =============================================================================

import io
import json
import os
import pickle
import warnings
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Optional

import holidays as holidays_lib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Rutas de datos (relativas al directorio de trabajo, que debe ser la raiz del repo)
# ---------------------------------------------------------------------------
MODEL_PATH    = "data/output/model_lgbm.pkl"
PREDS_PATH    = "data/output/predictions.csv"
TRUSTED_PATH  = "data/processed/trusted.parquet"
ENCODERS_PATH = "data/processed/label_encoders.json"
METRICS_PATH  = "data/output/metrics.json"

S3_BUCKET = "metro-medellin-datalake"
S3_KEY    = "models/model_lgbm.pkl"


# ---------------------------------------------------------------------------
# Estado global de la aplicacion (cargado una sola vez en startup)
# ---------------------------------------------------------------------------
state: dict = {}


def _cargar_modelo_desde_s3() -> object:
    """Intenta descargar model_lgbm.pkl desde S3 usando boto3."""
    import boto3
    s3 = boto3.client("s3")
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, S3_KEY, buf)
    buf.seek(0)
    return pickle.load(buf)


def _cargar_modelo() -> object:
    """
    Carga el modelo LightGBM con fallback:
      1. S3 si METRO_MODE=aws
      2. data/output/model_lgbm.pkl local
    """
    if os.environ.get("METRO_MODE") == "aws":
        try:
            print("[API] METRO_MODE=aws — descargando modelo desde S3...")
            modelo = _cargar_modelo_desde_s3()
            print("[API] Modelo cargado desde S3 correctamente")
            return modelo
        except Exception as exc:
            print(f"[API] Fallo S3 ({exc}), usando modelo local como fallback")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontro el modelo en '{MODEL_PATH}'. "
            "Ejecuta el pipeline primero o configura METRO_MODE=aws."
        )
    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)
    print(f"[API] Modelo cargado desde disco: {MODEL_PATH}")
    return modelo


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga todos los recursos pesados una sola vez al arrancar la API."""
    print("[API] Inicializando recursos...")

    state["modelo"] = _cargar_modelo()

    state["predictions"] = pd.read_csv(PREDS_PATH, parse_dates=["fecha"])
    # Normalizar columna linea para busquedas case-insensitive
    state["predictions"]["linea_norm"] = state["predictions"]["linea"].str.upper().str.strip()
    print(f"[API] predictions.csv cargado: {len(state['predictions']):,} filas")

    trusted = pd.read_parquet(TRUSTED_PATH)
    trusted["fecha"] = pd.to_datetime(trusted["fecha"])
    # Precalcular dia_semana para el fallback de lags
    trusted["dia_semana"] = trusted["fecha"].dt.dayofweek
    state["trusted"] = trusted
    print(f"[API] trusted.parquet cargado: {len(state['trusted']):,} filas")

    with open(ENCODERS_PATH, "r", encoding="utf-8") as f:
        state["encoders"] = json.load(f)
    print(f"[API] Encoders cargados: {list(state['encoders'].keys())}")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        state["metrics"] = json.load(f)
    print(f"[API] Metricas cargadas: {list(state['metrics'].keys())}")

    # Tabla de lineas disponibles (linea → tipo_linea)
    df_trusted = state["trusted"]
    state["lineas"] = (
        df_trusted[["linea", "tipo_linea"]]
        .drop_duplicates()
        .sort_values("linea")
        .to_dict(orient="records")
    )
    print(f"[API] Lineas disponibles: {len(state['lineas'])}")

    # Nombres exactos de features que el modelo LightGBM espera
    try:
        state["feature_cols"] = state["modelo"].feature_name_
    except AttributeError:
        # Fallback: derivar desde trusted.parquet (misma logica que 03_models.py)
        COLS_EXCLUIR = {"fecha", "DIA_raw", "linea", "tipo_linea", "hora", "pasajeros", "TOTAL"}
        state["feature_cols"] = [c for c in df_trusted.columns if c not in COLS_EXCLUIR]
    print(f"[API] Features del modelo: {len(state['feature_cols'])} columnas")

    print("[API] Listo para recibir peticiones")
    yield
    # Cleanup (liberar memoria si es necesario)
    state.clear()


# ---------------------------------------------------------------------------
# Aplicacion FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Prediccion de Afluencia — Metro de Medellin",
    description=(
        "Sirve predicciones de pasajeros por hora, linea y fecha usando "
        "un modelo LightGBM entrenado sobre datos 2023-2024."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class LookupRequest(BaseModel):
    linea: str = Field(..., example="LINEA A")
    fecha: str = Field(..., example="2025-03-15")
    hora: int = Field(..., ge=0, le=23, example=17)
    modelo: Optional[str] = Field("XGBoost", example="XGBoost")


class LookupResponse(BaseModel):
    linea: str
    fecha: str
    hora: int
    prediccion: float
    real: Optional[float]
    error_absoluto: Optional[float]
    modelo: str
    fuente: str


class LiveRequest(BaseModel):
    linea: str = Field(..., example="LINEA A")
    fecha: str = Field(..., example="2025-06-02")
    hora: int = Field(..., ge=0, le=23, example=17)


class LiveResponse(BaseModel):
    linea: str
    fecha: str
    hora: int
    prediccion: float
    modelo: str
    fuente: str
    lags_disponibles: bool


# ---------------------------------------------------------------------------
# Helpers para /predict/live
# ---------------------------------------------------------------------------
def _get_tipo_linea(linea: str) -> str:
    """Retorna el tipo_linea de una linea buscando en trusted.parquet."""
    df = state["trusted"]
    fila = df[df["linea"].str.upper().str.strip() == linea.upper().strip()]
    if fila.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Linea '{linea}' no encontrada en los datos historicos.",
        )
    return fila["tipo_linea"].iloc[0]


def _buscar_lag(df: pd.DataFrame, linea: str, hora: int, fecha_objetivo: date) -> Optional[float]:
    """Busca pasajeros de linea+hora en una fecha exacta en trusted.parquet."""
    sub = df[
        (df["linea"].str.upper().str.strip() == linea.upper().strip())
        & (df["hora"] == hora)
        & (df["fecha"].dt.date == fecha_objetivo)
    ]
    if sub.empty:
        return None
    return float(sub["pasajeros"].iloc[0])


def _promedio_historico(df: pd.DataFrame, linea: str, hora: int, dia_semana: int) -> float:
    """Promedio historico de pasajeros para linea+hora+dia_semana (fallback de lags)."""
    sub = df[
        (df["linea"].str.upper().str.strip() == linea.upper().strip())
        & (df["hora"] == hora)
        & (df["dia_semana"] == dia_semana)
    ]
    if sub.empty or sub["pasajeros"].isna().all():
        return 0.0
    return float(sub["pasajeros"].mean())


def _rolling_stats(df: pd.DataFrame, linea: str, hora: int, fecha_objetivo: date):
    """Calcula rolling_7d, rolling_28d y rolling_7d_std con datos anteriores a fecha_objetivo."""
    sub = df[
        (df["linea"].str.upper().str.strip() == linea.upper().strip())
        & (df["hora"] == hora)
        & (df["fecha"].dt.date < fecha_objetivo)
    ].sort_values("fecha")

    pax = sub["pasajeros"].dropna().values
    r7   = float(np.mean(pax[-7:]))  if len(pax) >= 3  else float(np.mean(pax)) if len(pax) else 0.0
    r28  = float(np.mean(pax[-28:])) if len(pax) >= 14 else float(np.mean(pax)) if len(pax) else 0.0
    r7s  = float(np.std(pax[-7:]))   if len(pax) >= 3  else 0.0
    return r7, r28, r7s


def _construir_features(linea: str, fecha_dt: pd.Timestamp, hora: int) -> pd.DataFrame:
    """
    Construye el DataFrame de una fila con todas las features que el modelo espera.
    Los nombres de columna coinciden con los generados en 02_features.py.
    """
    df_trusted = state["trusted"]
    encoders   = state["encoders"]
    festivos_col = holidays_lib.Colombia(years=[fecha_dt.year])
    fechas_festivas = set(festivos_col.keys())
    fechas_puente   = {f for f in fechas_festivas if pd.Timestamp(f).dayofweek == 0}

    fecha_date  = fecha_dt.date()
    dia_semana  = fecha_dt.dayofweek
    mes         = fecha_dt.month
    ano         = fecha_dt.year
    dia_del_ano = fecha_dt.dayofyear
    semana_del_ano = int(fecha_dt.isocalendar()[1])
    es_festivo  = int(fecha_date in fechas_festivas)
    es_puente   = int(fecha_date in fechas_puente)
    es_fin_de_semana = int(dia_semana >= 5)
    ano_trend   = ano - 2023

    # Columnas ciclicas
    hora_sin        = np.sin(2 * np.pi * hora / 24)
    hora_cos        = np.cos(2 * np.pi * hora / 24)
    dia_semana_sin  = np.sin(2 * np.pi * dia_semana / 7)
    dia_semana_cos  = np.cos(2 * np.pi * dia_semana / 7)
    mes_sin         = np.sin(2 * np.pi * mes / 12)
    mes_cos         = np.cos(2 * np.pi * mes / 12)

    # Encoding de linea
    linea_map      = encoders["linea"]
    tipo_linea_map = encoders["tipo_linea"]

    linea_norm = linea.upper().strip()
    # Buscar la clave en el mapa (puede venir con acento, p.e. "LÍNEA A")
    linea_key = next((k for k in linea_map if k.upper().strip() == linea_norm), None)
    if linea_key is None:
        raise HTTPException(
            status_code=422,
            detail=f"Linea '{linea}' no reconocida. Disponibles: {list(linea_map.keys())}",
        )

    tipo_linea_str  = _get_tipo_linea(linea)
    linea_encoded      = linea_map[linea_key]
    tipo_linea_encoded = tipo_linea_map.get(tipo_linea_str, 0)
    linea_es_metro_ferreo = int(tipo_linea_str == "metro_ferreo")

    # Interacciones
    hora_x_dia_semana   = hora * dia_semana
    es_hora_pico_manana = int(6 <= hora <= 9)
    es_hora_pico_tarde  = int(17 <= hora <= 19)

    # Lags (busca en trusted.parquet)
    lag1  = _buscar_lag(df_trusted, linea_key, hora, fecha_date - timedelta(days=1))
    lag7  = _buscar_lag(df_trusted, linea_key, hora, fecha_date - timedelta(days=7))
    lag14 = _buscar_lag(df_trusted, linea_key, hora, fecha_date - timedelta(days=14))
    lags_disponibles = all(v is not None for v in [lag1, lag7, lag14])

    if not lags_disponibles:
        fallback = _promedio_historico(df_trusted, linea_key, hora, dia_semana)
        if lag1  is None: lag1  = fallback
        if lag7  is None: lag7  = fallback
        if lag14 is None: lag14 = fallback

    r7, r28, r7s = _rolling_stats(df_trusted, linea_key, hora, fecha_date)

    # Construir dict con nombres exactos usados en 02_features.py
    # Las columnas con 'ñ' se usan si el modelo fue entrenado con ellas;
    # si feature_cols no las contiene, se usan los nombres sin acento.
    feature_cols = state["feature_cols"]

    # Determinar si el parquet usa 'año' o 'ano'
    usa_tilde = any("ñ" in c for c in feature_cols)

    row = {
        "hora_del_dia":          hora,
        "dia_semana":            dia_semana,
        "mes":                   mes,
        "año" if usa_tilde else "ano": ano,
        "dia_del_año" if usa_tilde else "dia_del_ano": dia_del_ano,
        "es_fin_de_semana":      es_fin_de_semana,
        "semana_del_año" if usa_tilde else "semana_del_ano": semana_del_ano,
        "es_festivo":            es_festivo,
        "es_puente":             es_puente,
        "hora_sin":              hora_sin,
        "hora_cos":              hora_cos,
        "dia_semana_sin":        dia_semana_sin,
        "dia_semana_cos":        dia_semana_cos,
        "mes_sin":               mes_sin,
        "mes_cos":               mes_cos,
        "linea_encoded":         linea_encoded,
        "tipo_linea_encoded":    tipo_linea_encoded,
        "lag_1d":                lag1,
        "lag_7d":                lag7,
        "lag_14d":               lag14,
        "rolling_7d":            r7,
        "rolling_28d":           r28,
        "rolling_7d_std":        r7s,
        "hora_x_dia_semana":     hora_x_dia_semana,
        "es_hora_pico_manana":   es_hora_pico_manana,
        "es_hora_pico_tarde":    es_hora_pico_tarde,
        "linea_es_metro_ferreo": linea_es_metro_ferreo,
        "año_trend" if usa_tilde else "ano_trend": ano_trend,
    }

    # Retener solo las columnas que el modelo conoce, en el orden correcto
    row_filtrado = {k: row[k] for k in feature_cols if k in row}
    return pd.DataFrame([row_filtrado])[feature_cols], lags_disponibles


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"])
def health():
    """Verifica que la API y el modelo estan operativos."""
    return {"status": "ok", "model": "LightGBM", "version": "1.0"}


@app.get("/lines", tags=["Informacion"])
def get_lines():
    """
    Retorna la lista de las 12 lineas del Metro de Medellin con su tipo.

    Tipos posibles: metro_ferreo, cable, tranvia, bus (Metroplus).
    """
    return {"lineas": state["lineas"], "total": len(state["lineas"])}


@app.post("/predict/lookup", response_model=LookupResponse, tags=["Prediccion"])
def predict_lookup(req: LookupRequest):
    """
    Busca en el archivo predictions.csv una prediccion ya calculada.

    Util para consultar resultados del test set 2025 sin reejecutar el modelo.
    El campo 'modelo' es opcional y filtra por el modelo deseado (default XGBoost).

    Retorna HTTP 404 si la combinacion linea+fecha+hora+modelo no existe.
    """
    df = state["predictions"]

    try:
        fecha_dt = pd.to_datetime(req.fecha).date()
    except Exception:
        raise HTTPException(status_code=422, detail=f"Formato de fecha invalido: '{req.fecha}'. Use YYYY-MM-DD.")

    mascara = (
        (df["linea_norm"] == req.linea.upper().strip())
        & (df["fecha"].dt.date == fecha_dt)
        & (df["hora"] == req.hora)
    )
    if req.modelo:
        mascara &= df["modelo"].str.strip() == req.modelo.strip()

    resultado = df[mascara]

    if resultado.empty:
        modelos_disp = df["modelo"].unique().tolist()
        raise HTTPException(
            status_code=404,
            detail=(
                f"No se encontro prediccion para linea='{req.linea}', "
                f"fecha='{req.fecha}', hora={req.hora}, modelo='{req.modelo}'. "
                f"Modelos disponibles: {modelos_disp}. "
                "Recuerde que predictions.csv solo cubre el año 2025."
            ),
        )

    fila = resultado.iloc[0]
    real_val = fila["real"] if not pd.isna(fila["real"]) else None
    err_val  = fila["error_absoluto"] if not pd.isna(fila["error_absoluto"]) else None

    return LookupResponse(
        linea=str(fila["linea"]),
        fecha=str(fila["fecha"].date()),
        hora=int(fila["hora"]),
        prediccion=round(float(fila["prediccion"]), 2),
        real=round(real_val, 2) if real_val is not None else None,
        error_absoluto=round(err_val, 2) if err_val is not None else None,
        modelo=str(fila["modelo"]),
        fuente="lookup",
    )


@app.post("/predict/live", response_model=LiveResponse, tags=["Prediccion"])
def predict_live(req: LiveRequest):
    """
    Calcula una prediccion en tiempo real usando el modelo LightGBM cargado.

    A diferencia de /predict/lookup, puede predecir cualquier fecha (pasada o futura)
    siempre que existan datos historicos para calcular los lags.

    Proceso interno:
      1. Extrae features temporales y festivos colombianos de la fecha.
      2. Busca lags (lag_1d, lag_7d, lag_14d) en trusted.parquet.
         Si no existen, usa el promedio historico linea+hora+dia_semana como fallback.
      3. Calcula rolling_7d/28d/std con datos anteriores a la fecha solicitada.
      4. Aplica label_encoders para linea y tipo_linea.
      5. Predice con LightGBM y aplica expm1() para invertir el log1p del entrenamiento.
    """
    try:
        fecha_dt = pd.Timestamp(req.fecha)
    except Exception:
        raise HTTPException(status_code=422, detail=f"Formato de fecha invalido: '{req.fecha}'. Use YYYY-MM-DD.")

    X, lags_disponibles = _construir_features(req.linea, fecha_dt, req.hora)

    modelo = state["modelo"]
    try:
        pred_log = modelo.predict(X)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {exc}")

    pred_pax = float(np.expm1(pred_log))
    pred_pax = max(pred_pax, 0.0)

    return LiveResponse(
        linea=req.linea.upper().strip(),
        fecha=str(fecha_dt.date()),
        hora=req.hora,
        prediccion=round(pred_pax, 1),
        modelo="LightGBM",
        fuente="live",
        lags_disponibles=lags_disponibles,
    )


@app.get("/metrics/summary", tags=["Metricas"])
def metrics_summary():
    """
    Retorna un resumen de las metricas globales de los modelos principales.

    Solo incluye las metricas agregadas (MAE, RMSE, R²) de cada modelo,
    sin el detalle por linea. Fuente: data/output/metrics.json.
    """
    metrics_raw = state["metrics"]
    MODELOS_PRINCIPALES = ["Baseline", "RidgeBaseline", "RandomForest", "LightGBM", "XGBoost"]

    resumen = {}
    for nombre in MODELOS_PRINCIPALES:
        if nombre in metrics_raw and "global" in metrics_raw[nombre]:
            g = metrics_raw[nombre]["global"]
            entrada = {
                "mae":  g.get("mae"),
                "rmse": g.get("rmse"),
                "r2":   g.get("r2"),
                "n":    g.get("n"),
            }
            if "mejora_mae_vs_baseline_pct" in metrics_raw[nombre]:
                entrada["mejora_mae_vs_baseline_pct"] = metrics_raw[nombre]["mejora_mae_vs_baseline_pct"]
                entrada["mejora_rmse_vs_baseline_pct"] = metrics_raw[nombre]["mejora_rmse_vs_baseline_pct"]
            resumen[nombre] = entrada

    return {
        "modelos": resumen,
        "periodo_test": "2025-01-01 / 2025-12-31",
        "periodo_train": "2023-01-01 / 2024-10-31",
    }
