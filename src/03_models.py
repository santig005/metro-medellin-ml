"""
Modelos de ML — Metro de Medellín
Baseline, RidgeBaseline, Random Forest, LightGBM global, LightGBM por línea, XGBoost.
Split temporal estricto: Train 2023-24, Test 2025.
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import holidays as holidays_lib
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

INPUT             = "data/processed/featured.parquet"
OUT_PRED          = "data/output/predictions.csv"
OUT_METRICS       = "data/output/metrics.json"
OUT_IMP           = "data/output/feature_importance.csv"
OUT_MODEL         = "data/output/model_lgbm.pkl"
OUT_MODEL_PERLINE = "data/output/models_per_line.pkl"  # dict con 12 modelos + función predict

# Features que el modelo NO debe ver nunca
COLS_EXCLUIR = {
    "fecha", "DIA_raw", "linea", "tipo_linea", "hora",
    "pasajeros", "TOTAL",
}

# Features categóricas para LightGBM (necesitan dtype int en pandas)
CAT_FEATURES = ["linea_encoded", "tipo_linea_encoded", "dia_semana", "mes", "hora_del_dia"]


# ---------------------------------------------------------------------------
# MÉTRICAS
# ---------------------------------------------------------------------------
def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}


def metricas_por_grupo(df_test: pd.DataFrame, pred_col: str, grupo: str) -> dict:
    """Calcula MAE/RMSE/R² para cada valor único del grupo."""
    resultado = {}
    for val, grp in df_test.groupby(grupo):
        y_real = grp["pasajeros"].values
        y_pred = grp[pred_col].values
        resultado[str(val)] = calcular_metricas(y_real, y_pred)
    return resultado


def metricas_por_festivo(df_test: pd.DataFrame, pred_col: str) -> dict:
    """
    Calcula MAE/RMSE para días festivos vs normales en el test set.
    Clasifica festivos en 'quieto' (demanda real < 60% media) y 'activo'.
    """
    festivos_col = holidays_lib.Colombia(years=[2025])
    fechas_festivas = set(festivos_col.keys())

    df = df_test.copy()
    df["es_festivo_calc"] = df["fecha"].dt.date.apply(lambda d: d in fechas_festivas).astype(int)

    # Demanda media de días normales (para clasificar festivos quiet/activo)
    media_normal = df[df["es_festivo_calc"] == 0]["pasajeros"].mean()

    resultados = {}
    for tipo in [0, 1]:
        label = "dias_normales" if tipo == 0 else "dias_festivos"
        sub = df[df["es_festivo_calc"] == tipo]
        if len(sub) < 5:
            continue
        m = calcular_metricas(sub["pasajeros"].values, sub[pred_col].values)
        m["n_registros"] = len(sub)
        m["n_dias"] = sub["fecha"].dt.date.nunique()
        resultados[label] = m

    # Desglose por día festivo individual
    por_dia = {}
    for fecha in sorted(df[df["es_festivo_calc"] == 1]["fecha"].dt.date.unique()):
        sub_f = df[df["fecha"].dt.date == fecha]
        nombre = festivos_col.get(fecha, str(fecha))
        demanda_real_pct = sub_f["pasajeros"].mean() / media_normal * 100
        tipo_festivo = "activo" if demanda_real_pct >= 60 else "quieto"
        m = calcular_metricas(sub_f["pasajeros"].values, sub_f[pred_col].values)
        m["nombre"] = nombre
        m["demanda_real_pct_vs_normal"] = round(demanda_real_pct, 1)
        m["tipo_festivo"] = tipo_festivo
        por_dia[str(fecha)] = m

    resultados["por_dia_festivo"] = por_dia

    # Resumen por tipo de festivo
    dias_quietos = [v for v in por_dia.values() if v["tipo_festivo"] == "quieto"]
    dias_activos = [v for v in por_dia.values() if v["tipo_festivo"] == "activo"]
    if dias_quietos:
        resultados["festivos_quietos"] = {
            "mae_media": round(sum(d["mae"] for d in dias_quietos) / len(dias_quietos), 2),
            "n_dias": len(dias_quietos),
        }
    if dias_activos:
        resultados["festivos_activos"] = {
            "mae_media": round(sum(d["mae"] for d in dias_activos) / len(dias_activos), 2),
            "n_dias": len(dias_activos),
        }
    return resultados


# ---------------------------------------------------------------------------
# TRANSFORMACIÓN LOG — normaliza la escala entre líneas de alta y baja demanda
# La diferencia 230× entre LÍNEA A y LÍNEA H haría que el modelo optimizara
# solo para las líneas grandes. log1p nivela el error relativo.
# ---------------------------------------------------------------------------
def log_transform(y: np.ndarray) -> np.ndarray:
    return np.log1p(y)

def log_inverse(y: np.ndarray) -> np.ndarray:
    return np.expm1(y)


# ---------------------------------------------------------------------------
# SPLIT TEMPORAL
# ---------------------------------------------------------------------------
def hacer_split(df: pd.DataFrame):
    # Descartar filas sin target o con lags NaN (primeras semanas del train)
    df = df.dropna(subset=["pasajeros", "lag_1d", "lag_7d", "rolling_7d"]).copy()

    mask_test  = df["fecha"].dt.year == 2025
    mask_train = ~mask_test

    df_train = df[mask_train].copy()
    df_test  = df[mask_test].copy()

    # Validación: últimos 2 meses del train (nov-dic 2024) para early stopping
    mask_val = (df_train["fecha"] >= "2024-11-01")
    df_val   = df_train[mask_val].copy()
    df_tr    = df_train[~mask_val].copy()

    print(f"[Split] Train efectivo : {len(df_tr):,}  "
          f"({df_tr['fecha'].min().date()} → {df_tr['fecha'].max().date()})")
    print(f"[Split] Validación     : {len(df_val):,}  "
          f"({df_val['fecha'].min().date()} → {df_val['fecha'].max().date()})")
    print(f"[Split] Test           : {len(df_test):,}  "
          f"({df_test['fecha'].min().date()} → {df_test['fecha'].max().date()})")

    return df_tr, df_val, df_train, df_test


def preparar_X_y(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].copy()
    y = df["pasajeros"].values
    return X, y


# ---------------------------------------------------------------------------
# MODELO 1 — BASELINE: promedio histórico por (linea, hora, dia_semana)
# ---------------------------------------------------------------------------
def entrenar_baseline(df_train: pd.DataFrame) -> dict:
    """Tabla de lookup: media por (linea, hora_del_dia, dia_semana) sobre train completo."""
    tabla = (
        df_train.groupby(["linea", "hora_del_dia", "dia_semana"])["pasajeros"]
        .mean()
        .rename("pred_baseline")
        .reset_index()
    )
    return tabla


def predecir_baseline(df_test: pd.DataFrame, tabla: dict) -> np.ndarray:
    merged = df_test[["linea", "hora_del_dia", "dia_semana"]].merge(
        tabla, on=["linea", "hora_del_dia", "dia_semana"], how="left"
    )
    # Para combinaciones no vistas usar la media global del train
    media_global = tabla["pred_baseline"].mean()
    pred = merged["pred_baseline"].fillna(media_global).values
    return pred


# ---------------------------------------------------------------------------
# MODELO 2 — RIDGE REGRESSION (baseline lineal)
# ---------------------------------------------------------------------------
def entrenar_ridge(X_train, y_train) -> Pipeline:
    print("[Ridge] Entrenando Ridge Regression (alpha=1.0, OHE + StandardScaler)...")
    # linea_encoded y tipo_linea_encoded son enteros 0-11 / 0-3, no ordinales.
    # Con label-encoding Ridge aprende un coeficiente lineal que no captura la
    # diferencia de escala 230× entre líneas. OHE da una intercepción por línea.
    cat_cols  = ["linea_encoded", "tipo_linea_encoded"]
    num_cols  = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    modelo = Pipeline([
        ("prep",  preprocessor),
        ("ridge", Ridge(alpha=1.0)),
    ])
    modelo.fit(X_train, y_train)
    print("[Ridge] Entrenamiento completado")
    return modelo


# ---------------------------------------------------------------------------
# MODELO 3 — RANDOM FOREST
# ---------------------------------------------------------------------------
def entrenar_rf(X_train, y_train, feature_cols: list) -> RandomForestRegressor:
    print("[RF] Entrenando Random Forest (n=200, depth=15)...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    print("[RF] Entrenamiento completado")
    return rf


# ---------------------------------------------------------------------------
# MODELO 3 — LIGHTGBM (modelo principal)
# ---------------------------------------------------------------------------
def entrenar_lgbm(X_tr, y_tr, X_val, y_val, feature_cols: list) -> lgb.LGBMRegressor:
    cat_valid = [c for c in CAT_FEATURES if c in feature_cols]
    print(f"[LGB] Entrenando LightGBM — features categóricas: {cat_valid}")

    modelo = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,     # más bajo para mayor precisión con log target
        num_leaves=255,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    modelo.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200),
        ],
        categorical_feature=cat_valid,
    )
    print(f"[LGB] Mejor iteración: {modelo.best_iteration_}")
    return modelo


# ---------------------------------------------------------------------------
# MODELO 4 — XGBOOST
# ---------------------------------------------------------------------------
def entrenar_xgb(X_tr, y_tr, X_val, y_val) -> xgb.XGBRegressor:
    print("[XGB] Entrenando XGBoost...")
    modelo = xgb.XGBRegressor(
        n_estimators=397,
        learning_rate=0.0995,
        max_depth=7,
        min_child_weight=10,
        subsample=0.961,
        colsample_bytree=0.713,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,
        eval_metric="mae",
        verbosity=0,
    )
    modelo.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"[XGB] Mejor iteración: {modelo.best_iteration}")
    return modelo


# ---------------------------------------------------------------------------
# EVALUACIÓN COMPLETA
# ---------------------------------------------------------------------------
def evaluar_modelo(
    nombre: str,
    df_test: pd.DataFrame,
    pred: np.ndarray,
    baseline_mae: float,
    baseline_rmse: float,
) -> dict:
    df_test = df_test.copy()
    df_test[f"pred_{nombre}"] = np.maximum(pred, 0)   # predicciones no pueden ser negativas

    metricas_globales = calcular_metricas(df_test["pasajeros"].values, df_test[f"pred_{nombre}"].values)
    por_linea    = metricas_por_grupo(df_test, f"pred_{nombre}", "linea")
    por_tipo     = metricas_por_grupo(df_test, f"pred_{nombre}", "tipo_linea")
    por_hora     = metricas_por_grupo(df_test, f"pred_{nombre}", "hora_del_dia")
    por_festivo  = metricas_por_festivo(df_test, f"pred_{nombre}")

    mejora_mae  = (baseline_mae  - metricas_globales["mae"])  / baseline_mae  * 100
    mejora_rmse = (baseline_rmse - metricas_globales["rmse"]) / baseline_rmse * 100

    resultado = {
        "global":       metricas_globales,
        "por_linea":    por_linea,
        "por_tipo":     por_tipo,
        "por_hora":     por_hora,
        "por_festivo":  por_festivo,
        "mejora_mae_vs_baseline_pct":  round(mejora_mae,  2),
        "mejora_rmse_vs_baseline_pct": round(mejora_rmse, 2),
    }
    return resultado, df_test


# ---------------------------------------------------------------------------
# TABLA COMPARATIVA IMPRESA
# ---------------------------------------------------------------------------
def imprimir_comparativa(todos_metricas: dict) -> None:
    print("\n" + "=" * 65)
    print("COMPARATIVA DE MODELOS — TEST SET 2025")
    print("=" * 65)
    print(f"{'Modelo':<20} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'%↓MAE':>8} {'%↓RMSE':>9}")
    print("-" * 65)
    for nombre, m in todos_metricas.items():
        g = m["global"]
        pmae  = m.get("mejora_mae_vs_baseline_pct",  0)
        prmse = m.get("mejora_rmse_vs_baseline_pct", 0)
        print(f"  {nombre:<18} {g['mae']:>10,.1f} {g['rmse']:>10,.1f} "
              f"{g['r2']:>8.4f} {pmae:>7.1f}% {prmse:>8.1f}%")
    print("=" * 65)


# ---------------------------------------------------------------------------
# MODELOS POR LÍNEA — 12 LightGBM individuales
# Se guardan en un dict serializable con una función predict() unificada.
# ---------------------------------------------------------------------------
def entrenar_lgbm_por_linea(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list,
) -> dict:
    """
    Entrena un LightGBM por cada línea sobre datos de train/validación.
    Retorna un dict con la estructura:
        {
            "modelos":   {linea: lgb.LGBMRegressor},
            "features":  feature_cols,
            "predict":   <función predict(linea, X_df) -> np.ndarray>,
        }
    """
    lineas = sorted(df_tr["linea"].unique())
    cat_valid = [c for c in CAT_FEATURES if c in feature_cols]
    modelos_dict = {}

    for linea in lineas:
        tr_l  = df_tr[df_tr["linea"] == linea]
        val_l = df_val[df_val["linea"] == linea]

        if len(tr_l) < 100 or len(val_l) < 10:
            print(f"  [PL] {linea}: datos insuficientes, omitido")
            continue

        X_tr_l,  y_tr_l  = preparar_X_y(tr_l,  feature_cols)
        X_val_l, y_val_l = preparar_X_y(val_l, feature_cols)

        y_tr_l  = log_transform(y_tr_l)
        y_val_l = log_transform(y_val_l)

        m = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=127,
            min_child_samples=5,    # menos muestras mínimas por hoja (líneas pequeñas)
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(
            X_tr_l, y_tr_l,
            eval_set=[(X_val_l, y_val_l)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=9999),  # silenciar iteraciones
            ],
            categorical_feature=cat_valid,
        )
        modelos_dict[linea] = m
        print(f"  [PL] {linea}: {len(tr_l):,} train  iter={m.best_iteration_}")

    # Función de predicción unificada — compatible con AWS EMR
    def predict(linea: str, X: pd.DataFrame) -> np.ndarray:
        if linea not in modelos_dict:
            raise KeyError(f"No hay modelo entrenado para '{linea}'")
        return log_inverse(np.maximum(modelos_dict[linea].predict(X[feature_cols]), 0))

    return {
        "modelos":  modelos_dict,
        "features": feature_cols,
        "predict":  predict,
    }


def predecir_per_line(bundle: dict, df_test: pd.DataFrame) -> np.ndarray:
    """Aplica cada modelo de línea a su subconjunto del test set."""
    pred = np.zeros(len(df_test))
    feature_cols = bundle["features"]
    for linea, modelo in bundle["modelos"].items():
        mask = (df_test["linea"] == linea).values
        if mask.sum() == 0:
            continue
        X_sub = df_test[feature_cols].iloc[mask]
        pred[mask] = log_inverse(np.maximum(modelo.predict(X_sub), 0))
    return pred


# ---------------------------------------------------------------------------
# REGENERACIÓN DE metrics.json DESDE predictions.csv
# Independiente del estado en memoria — garantiza JSON válido al final del pipeline.
# ---------------------------------------------------------------------------
def regenerar_metrics_desde_predicciones(pred_path: str, featured_path: str, metrics_path: str) -> dict:
    """
    Lee predictions.csv y featured.parquet, recalcula todas las métricas y
    sobreescribe metrics.json. Verifica la integridad con json.loads al final.
    """
    print(f"\n[Metricas] Regenerando {metrics_path} desde {pred_path}...")
    pred_df = pd.read_csv(pred_path, parse_dates=["fecha"])

    # Traer es_festivo del featured — merge por fecha + linea + hora
    feat_cols = ["fecha", "linea", "hora_del_dia", "es_festivo"]
    feat_df = pd.read_parquet(featured_path, columns=feat_cols)
    feat_df["fecha"] = pd.to_datetime(feat_df["fecha"])
    feat_df = feat_df.rename(columns={"hora_del_dia": "hora"})

    pred_df = pred_df.merge(feat_df, on=["fecha", "linea", "hora"], how="left")

    modelos = list(pred_df["modelo"].unique())
    base_sub = pred_df[pred_df["modelo"] == "Baseline"]
    baseline_mae  = mean_absolute_error(base_sub["real"], base_sub["prediccion"])
    baseline_rmse = np.sqrt(mean_squared_error(base_sub["real"], base_sub["prediccion"]))

    resultado = {}
    for modelo in modelos:
        sub = pred_df[pred_df["modelo"] == modelo].copy()
        y_real = sub["real"].values
        y_pred = sub["prediccion"].values

        mae  = mean_absolute_error(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        r2   = r2_score(y_real, y_pred)

        por_linea = {}
        for linea, grp in sub.groupby("linea"):
            por_linea[str(linea)] = {
                "mae":  round(float(mean_absolute_error(grp["real"], grp["prediccion"])), 2),
                "rmse": round(float(np.sqrt(mean_squared_error(grp["real"], grp["prediccion"]))), 2),
            }

        festivos = {}
        if "es_festivo" in sub.columns and sub["es_festivo"].notna().any():
            for tipo_val, label in [(1, "festivo"), (0, "normal")]:
                s = sub[sub["es_festivo"] == tipo_val]
                if len(s) >= 5:
                    festivos[label] = {
                        "mae":  round(float(mean_absolute_error(s["real"], s["prediccion"])), 2),
                        "rmse": round(float(np.sqrt(mean_squared_error(s["real"], s["prediccion"]))), 2),
                        "n":    int(len(s)),
                    }

        mejora_mae  = (baseline_mae  - mae)  / baseline_mae  * 100
        mejora_rmse = (baseline_rmse - rmse) / baseline_rmse * 100

        resultado[modelo] = {
            "global": {
                "mae":  round(float(mae),  2),
                "rmse": round(float(rmse), 2),
                "r2":   round(float(r2),   4),
                "n":    int(len(sub)),
            },
            "por_linea": por_linea,
            "festivos":  festivos,
            "mejora_mae_vs_baseline_pct":  round(float(mejora_mae),  2),
            "mejora_rmse_vs_baseline_pct": round(float(mejora_rmse), 2),
        }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    # Verificación de integridad
    json.loads(open(metrics_path, encoding="utf-8").read())
    print(f"[Metricas] JSON verificado. Modelos: {list(resultado.keys())}")
    for nombre, m in resultado.items():
        print(f"  {nombre:<20} MAE={m['global']['mae']:>8,.1f}  "
              f"RMSE={m['global']['rmse']:>8,.1f}  R²={m['global']['r2']:.4f}")
    return resultado


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs("data/output", exist_ok=True)

    print("[Modelos] Cargando featured.parquet...")
    df = pd.read_parquet(INPUT)
    print(f"[Modelos] Registros totales: {len(df):,}")

    # Determinar feature columns dinámicamente
    feature_cols = [c for c in df.columns if c not in COLS_EXCLUIR]
    print(f"[Modelos] Features ({len(feature_cols)}): {feature_cols}")

    # Split
    df_tr, df_val, df_train, df_test = hacer_split(df)

    X_tr,    y_tr_raw    = preparar_X_y(df_tr,    feature_cols)
    X_val,   y_val_raw   = preparar_X_y(df_val,   feature_cols)
    X_train, y_train_raw = preparar_X_y(df_train, feature_cols)
    X_test,  y_test      = preparar_X_y(df_test,  feature_cols)

    # Targets en escala log para entrenamiento de modelos
    y_tr    = log_transform(y_tr_raw)
    y_val   = log_transform(y_val_raw)
    y_train = log_transform(y_train_raw)

    print(f"\n[Modelos] Shape X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    todos_metricas = {}
    pred_dfs = []

    # --- BASELINE ---
    print("\n[Baseline] Calculando tabla de promedios históricos...")
    tabla_baseline = entrenar_baseline(df_train)
    pred_base = predecir_baseline(df_test, tabla_baseline)
    pred_base = np.maximum(pred_base, 0)
    m_base = calcular_metricas(y_test, pred_base)
    df_base_aux = df_test.assign(pred_baseline=pred_base)
    todos_metricas["Baseline"] = {
        "global": m_base,
        "por_linea": metricas_por_grupo(df_base_aux, "pred_baseline", "linea"),
        "por_tipo":  metricas_por_grupo(df_base_aux, "pred_baseline", "tipo_linea"),
        "por_hora":  metricas_por_grupo(df_base_aux, "pred_baseline", "hora_del_dia"),
        "por_festivo": metricas_por_festivo(df_base_aux, "pred_baseline"),
        "mejora_mae_vs_baseline_pct":  0.0,
        "mejora_rmse_vs_baseline_pct": 0.0,
    }
    print(f"[Baseline] MAE={m_base['mae']:,.1f}  RMSE={m_base['rmse']:,.1f}  R²={m_base['r2']:.4f}")
    baseline_mae  = m_base["mae"]
    baseline_rmse = m_base["rmse"]

    # Guardar predicciones baseline
    df_pred_base = df_test[["fecha", "linea", "hora_del_dia", "pasajeros"]].copy()
    df_pred_base["prediccion"] = pred_base
    df_pred_base["error_absoluto"] = (df_pred_base["pasajeros"] - df_pred_base["prediccion"]).abs()
    df_pred_base["modelo"] = "Baseline"
    pred_dfs.append(df_pred_base)

    # --- RIDGE BASELINE ---
    ridge = entrenar_ridge(X_train, y_train)
    pred_ridge = log_inverse(np.maximum(ridge.predict(X_test), 0))
    m_ridge, _ = evaluar_modelo("Ridge", df_test, pred_ridge, baseline_mae, baseline_rmse)
    todos_metricas["RidgeBaseline"] = m_ridge
    print(f"[Ridge] MAE={m_ridge['global']['mae']:,.1f}  RMSE={m_ridge['global']['rmse']:,.1f}  R²={m_ridge['global']['r2']:.4f}")

    df_pred_ridge = df_test[["fecha", "linea", "hora_del_dia", "pasajeros"]].copy()
    df_pred_ridge["prediccion"] = pred_ridge
    df_pred_ridge["error_absoluto"] = (df_pred_ridge["pasajeros"] - df_pred_ridge["prediccion"]).abs()
    df_pred_ridge["modelo"] = "RidgeBaseline"
    pred_dfs.append(df_pred_ridge)

    # --- RANDOM FOREST ---
    rf = entrenar_rf(X_train, y_train, feature_cols)
    pred_rf = log_inverse(np.maximum(rf.predict(X_test), 0))
    m_rf, df_test_rf = evaluar_modelo("RF", df_test, pred_rf, baseline_mae, baseline_rmse)
    todos_metricas["RandomForest"] = m_rf
    print(f"[RF] MAE={m_rf['global']['mae']:,.1f}  RMSE={m_rf['global']['rmse']:,.1f}  R²={m_rf['global']['r2']:.4f}")

    df_pred_rf = df_test[["fecha", "linea", "hora_del_dia", "pasajeros"]].copy()
    df_pred_rf["prediccion"] = pred_rf
    df_pred_rf["error_absoluto"] = (df_pred_rf["pasajeros"] - df_pred_rf["prediccion"]).abs()
    df_pred_rf["modelo"] = "RandomForest"
    pred_dfs.append(df_pred_rf)

    # --- LIGHTGBM ---
    lgbm = entrenar_lgbm(X_tr, y_tr, X_val, y_val, feature_cols)
    pred_lgb = log_inverse(np.maximum(lgbm.predict(X_test), 0))
    m_lgb, df_test_lgb = evaluar_modelo("LGBM", df_test, pred_lgb, baseline_mae, baseline_rmse)
    todos_metricas["LightGBM"] = m_lgb
    print(f"[LGB] MAE={m_lgb['global']['mae']:,.1f}  RMSE={m_lgb['global']['rmse']:,.1f}  R²={m_lgb['global']['r2']:.4f}")

    df_pred_lgb = df_test[["fecha", "linea", "hora_del_dia", "pasajeros"]].copy()
    df_pred_lgb["prediccion"] = pred_lgb
    df_pred_lgb["error_absoluto"] = (df_pred_lgb["pasajeros"] - df_pred_lgb["prediccion"]).abs()
    df_pred_lgb["modelo"] = "LightGBM"
    pred_dfs.append(df_pred_lgb)

    # Guardar modelo LightGBM
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(lgbm, f)
    print(f"[LGB] Modelo guardado en {OUT_MODEL}")

    # Feature importance LightGBM
    imp_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": lgbm.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(OUT_IMP, index=False)
    print(f"[LGB] Feature importance guardada en {OUT_IMP}")
    print("\n  Top 10 features LightGBM:")
    print(imp_df.head(10).to_string(index=False))

    # --- XGBOOST ---
    modelo_xgb = entrenar_xgb(X_tr, y_tr, X_val, y_val)
    pred_xgb_vals = log_inverse(np.maximum(modelo_xgb.predict(X_test), 0))
    m_xgb, _ = evaluar_modelo("XGB", df_test, pred_xgb_vals, baseline_mae, baseline_rmse)
    todos_metricas["XGBoost"] = m_xgb
    print(f"[XGB] MAE={m_xgb['global']['mae']:,.1f}  RMSE={m_xgb['global']['rmse']:,.1f}  R²={m_xgb['global']['r2']:.4f}")

    df_pred_xgb = df_test[["fecha", "linea", "hora_del_dia", "pasajeros"]].copy()
    df_pred_xgb["prediccion"] = pred_xgb_vals
    df_pred_xgb["error_absoluto"] = (df_pred_xgb["pasajeros"] - df_pred_xgb["prediccion"]).abs()
    df_pred_xgb["modelo"] = "XGBoost"
    pred_dfs.append(df_pred_xgb)

    # --- LIGHTGBM POR LÍNEA ---
    print("\n[PerLine] Entrenando 12 modelos LightGBM (uno por línea)...")
    bundle_pl = entrenar_lgbm_por_linea(df_tr, df_val, feature_cols)
    pred_pl = predecir_per_line(bundle_pl, df_test)
    m_pl, _ = evaluar_modelo("LGB_perline", df_test, pred_pl, baseline_mae, baseline_rmse)
    todos_metricas["LGB_PerLinea"] = m_pl
    print(f"[PerLine] MAE={m_pl['global']['mae']:,.1f}  RMSE={m_pl['global']['rmse']:,.1f}  R²={m_pl['global']['r2']:.4f}")
    print(f"[PerLine] Mejora vs baseline: MAE={m_pl['mejora_mae_vs_baseline_pct']:+.1f}%  "
          f"RMSE={m_pl['mejora_rmse_vs_baseline_pct']:+.1f}%")

    df_pred_pl = df_test[["fecha", "linea", "hora_del_dia", "pasajeros"]].copy()
    df_pred_pl["prediccion"] = pred_pl
    df_pred_pl["error_absoluto"] = (df_pred_pl["pasajeros"] - df_pred_pl["prediccion"]).abs()
    df_pred_pl["modelo"] = "LGB_PerLinea"
    pred_dfs.append(df_pred_pl)

    # Serializar bundle sin la función lambda (pickle serializa los modelos)
    bundle_serial = {
        "modelos":   bundle_pl["modelos"],
        "features":  bundle_pl["features"],
    }
    with open(OUT_MODEL_PERLINE, "wb") as f:
        pickle.dump(bundle_serial, f)
    print(f"[PerLine] Modelos guardados en {OUT_MODEL_PERLINE}")

    # ABLATION: ¿dia_del_año causa leakage estacional?
    # Solo corre si el modelo per-line NO supera al baseline
    if m_pl["global"]["mae"] > baseline_mae:
        print("\n[Ablation] LGB_PerLinea no supera baseline, corriendo test sin dia_del_año...")
        feat_sin_dia = [c for c in feature_cols if c != "dia_del_año"]
        X_tr_abl,  y_tr_abl  = preparar_X_y(df_tr,  feat_sin_dia)
        X_val_abl, y_val_abl = preparar_X_y(df_val, feat_sin_dia)
        X_test_abl,_         = preparar_X_y(df_test, feat_sin_dia)
        y_tr_abl  = log_transform(y_tr_abl)
        y_val_abl = log_transform(y_val_abl)
        lgbm_abl = entrenar_lgbm(X_tr_abl, y_tr_abl, X_val_abl, y_val_abl, feat_sin_dia)
        pred_abl = log_inverse(np.maximum(lgbm_abl.predict(X_test_abl), 0))
        m_abl, _ = evaluar_modelo("LGB_sinDiaAno", df_test, pred_abl, baseline_mae, baseline_rmse)
        todos_metricas["LGB_sinDiaAnio"] = m_abl
        print(f"[Ablation] Sin dia_del_año: MAE={m_abl['global']['mae']:,.1f}  "
              f"RMSE={m_abl['global']['rmse']:,.1f}  R²={m_abl['global']['r2']:.4f}")
        diff_mae = m_abl["global"]["mae"] - m_lgb["global"]["mae"]
        print(f"[Ablation] Diferencia vs LightGBM global: MAE {diff_mae:+.1f}  "
              f"({'peor' if diff_mae > 0 else 'mejor'} sin dia_del_año)")

    # --- GUARDAR PREDICCIONES ---
    pred_all = pd.concat(pred_dfs, ignore_index=True)
    pred_all = pred_all.rename(columns={"hora_del_dia": "hora", "pasajeros": "real"})
    pred_all.to_csv(OUT_PRED, index=False)
    print(f"\n[Modelos] Predicciones guardadas en {OUT_PRED}  ({len(pred_all):,} filas)")

    # --- REGENERAR metrics.json DESDE predictions.csv ---
    # Se regenera independientemente para evitar truncamiento del objeto en memoria.
    todos_metricas = regenerar_metrics_desde_predicciones(OUT_PRED, INPUT, OUT_METRICS)

    # Tabla final
    imprimir_comparativa(todos_metricas)

    # Resumen per-line
    print("\n  MAE por línea — comparativa Baseline vs LGB_PerLinea:")
    print(f"  {'Línea':<15} {'Baseline MAE':>14} {'PerLine MAE':>12} {'Mejora':>9}")
    print("  " + "-" * 55)
    for linea in sorted(todos_metricas["Baseline"]["por_linea"].keys()):
        mae_b  = todos_metricas["Baseline"]["por_linea"][linea]["mae"]
        mae_pl = todos_metricas["LGB_PerLinea"]["por_linea"].get(linea, {}).get("mae", float("nan"))
        mejora = (mae_b - mae_pl) / mae_b * 100 if mae_pl == mae_pl else float("nan")
        print(f"  {linea:<15} {mae_b:>14,.1f} {mae_pl:>12,.1f} {mejora:>8.1f}%")


# ---------------------------------------------------------------------------
# WALK-FORWARD CROSS-VALIDATION (PoC)
# Expanding window sobre datos 2023-2024. No modifica predictions.csv ni metrics.json.
# Ejecutar en aislamiento: python src/03_models.py --cv
# ---------------------------------------------------------------------------
def walkforward_cv():
    """
    4 folds de expanding window sobre el período de entrenamiento (2023-2024).
    Evalúa Baseline y LightGBM. No escribe ningún archivo de salida.

    Criterio de permanencia:
      - std(MAE LightGBM) < 15% de la media → modelo estable → mantener función
      - std > 15% o no aporta nada nuevo → eliminar
    """
    FOLDS = [
        ("Fold 1", "2023-01-01", "2023-06-30", "2023-07-01", "2023-09-30"),
        ("Fold 2", "2023-01-01", "2023-12-31", "2024-01-01", "2024-03-31"),
        ("Fold 3", "2023-01-01", "2024-06-30", "2024-07-01", "2024-09-30"),
        ("Fold 4", "2023-01-01", "2024-09-30", "2024-10-01", "2024-12-31"),
    ]
    VAL_FRACTION = 0.20   # último 20% del training de cada fold → early stopping

    print("\n" + "=" * 65)
    print("WALK-FORWARD CROSS-VALIDATION — PoC")
    print("Baseline histórico vs LightGBM  |  4 folds expanding window")
    print("=" * 65)

    print("[WF-CV] Cargando featured.parquet...")
    df = pd.read_parquet(INPUT)
    df = df.dropna(subset=["pasajeros", "lag_1d", "lag_7d", "rolling_7d"]).copy()
    feature_cols = [c for c in df.columns if c not in COLS_EXCLUIR]

    resultados = []

    for nombre, tr_ini, tr_fin, te_ini, te_fin in FOLDS:
        mask_tr = (df["fecha"] >= tr_ini) & (df["fecha"] <= tr_fin)
        mask_te = (df["fecha"] >= te_ini) & (df["fecha"] <= te_fin)

        df_tr_full = df[mask_tr].copy()
        df_te      = df[mask_te].copy()

        if len(df_tr_full) < 500 or len(df_te) < 100:
            print(f"  {nombre}: datos insuficientes, omitido")
            continue

        # Último 20% del training como validación para early stopping
        cut = int(len(df_tr_full) * (1 - VAL_FRACTION))
        df_tr  = df_tr_full.iloc[:cut]
        df_val = df_tr_full.iloc[cut:]

        X_tr,  y_tr_raw  = preparar_X_y(df_tr,  feature_cols)
        X_val, y_val_raw = preparar_X_y(df_val, feature_cols)
        X_te,  y_te      = preparar_X_y(df_te,  feature_cols)
        y_tr  = log_transform(y_tr_raw)
        y_val = log_transform(y_val_raw)

        # --- Baseline ---
        tabla_b = entrenar_baseline(df_tr_full)       # entrena sobre train completo del fold
        pred_b  = np.maximum(predecir_baseline(df_te, tabla_b), 0)
        mae_b   = mean_absolute_error(y_te, pred_b)

        # --- LightGBM ---
        lgbm_fold = entrenar_lgbm(X_tr, y_tr, X_val, y_val, feature_cols)
        pred_l    = log_inverse(np.maximum(lgbm_fold.predict(X_te), 0))
        mae_l     = mean_absolute_error(y_te, pred_l)

        lgbm_supera = "SI" if mae_l < mae_b else "no"

        print(f"\n  {nombre}  |  train: {tr_ini}→{tr_fin} ({len(df_tr_full):,} reg)"
              f"  |  test: {te_ini}→{te_fin} ({len(df_te):,} reg)")
        print(f"    Baseline  MAE = {mae_b:,.1f}")
        print(f"    LightGBM  MAE = {mae_l:,.1f}   "
              f"({'LGB gana' if mae_l < mae_b else 'Baseline gana'})")

        resultados.append({"fold": nombre, "mae_base": mae_b, "mae_lgb": mae_l,
                           "lgb_supera": lgbm_supera})

    if not resultados:
        print("\n[WF-CV] Sin resultados — todos los folds omitidos.")
        return

    maes_base = [r["mae_base"] for r in resultados]
    maes_lgb  = [r["mae_lgb"]  for r in resultados]

    media_b  = float(np.mean(maes_base));  std_b  = float(np.std(maes_base))
    media_l  = float(np.mean(maes_lgb));   std_l  = float(np.std(maes_lgb))
    cv_l     = std_l / media_l * 100   # coeficiente de variación LightGBM

    print("\n" + "-" * 65)
    print("RESUMEN WALK-FORWARD")
    print(f"  {'Modelo':<14} {'Media MAE':>10} {'Std MAE':>10} {'CV%':>8}")
    print(f"  {'Baseline':<14} {media_b:>10,.1f} {std_b:>10,.1f} {std_b/media_b*100:>7.1f}%")
    print(f"  {'LightGBM':<14} {media_l:>10,.1f} {std_l:>10,.1f} {cv_l:>7.1f}%")

    folds_lgb_gana = sum(1 for r in resultados if r["lgb_supera"] == "SI")
    print(f"\n  LightGBM supera a Baseline en {folds_lgb_gana}/{len(resultados)} folds")

    # --- Criterio de permanencia ---
    print("\n" + "-" * 65)
    UMBRAL_CV = 15.0
    if cv_l < UMBRAL_CV:
        print(f"  DECISION: CV LightGBM = {cv_l:.1f}% < {UMBRAL_CV}% → modelo ESTABLE")
        print("  → Función walkforward_cv() MANTENIDA en el código.")
    else:
        print(f"  DECISION: CV LightGBM = {cv_l:.1f}% >= {UMBRAL_CV}% → alta variabilidad")
        print("  → Evaluar si la función aporta valor adicional al split principal.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    if "--cv" in sys.argv:
        walkforward_cv()
    else:
        main()
