"""
Hyperparameter tuning — LightGBM y XGBoost
Rama: experiment/hyperparameter-tuning
NO modifica el pipeline principal ni escribe en data/output/.

Objetivo Optuna: MAE promedio sobre 2 folds de expanding window
  (misma lógica que walkforward_cv en src/03_models.py, Fold 3 + Fold 4)
  Fold 3: train 2023-01-01→2024-06-30  /  val 2024-07-01→2024-09-30
  Fold 4: train 2023-01-01→2024-09-30  /  val 2024-10-01→2024-12-31

Evaluación final: test set 2025 completo (igual que el pipeline).
"""

import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Silenciar logs verbosos
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).parent.parent
INPUT   = ROOT / "data/processed/featured.parquet"

# ---------------------------------------------------------------------------
# Constantes del pipeline (espejo de src/03_models.py para no importar el módulo)
# ---------------------------------------------------------------------------
COLS_EXCLUIR = {"fecha", "DIA_raw", "linea", "tipo_linea", "hora", "pasajeros", "TOTAL"}
CAT_FEATURES = ["linea_encoded", "tipo_linea_encoded", "dia_semana", "mes", "hora_del_dia"]

# Folds expanding window (mismos que walkforward_cv, usamos Fold3 + Fold4)
FOLDS_OPTUNA = [
    ("Fold 3", "2023-01-01", "2024-06-30", "2024-07-01", "2024-09-30"),
    ("Fold 4", "2023-01-01", "2024-09-30", "2024-10-01", "2024-12-31"),
]

# Defaults actuales del pipeline (para comparar al final)
LGBM_DEFAULTS = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=255,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1,
)
XGB_DEFAULTS = dict(
    n_estimators=2000, learning_rate=0.02, max_depth=8, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    eval_metric="mae", verbosity=0,
)

# ---------------------------------------------------------------------------
# Utilidades (mismas que en 03_models.py)
# ---------------------------------------------------------------------------
def log_transform(y):  return np.log1p(y)
def log_inverse(y):    return np.expm1(y)

def preparar_X_y(df, feature_cols):
    return df[feature_cols].copy(), df["pasajeros"].values


def cargar_datos():
    """Carga featured.parquet y devuelve df limpio + feature_cols."""
    print(f"[Tuning] Cargando {INPUT} ...")
    df = pd.read_parquet(INPUT)
    df = df.dropna(subset=["pasajeros", "lag_1d", "lag_7d", "rolling_7d"]).copy()
    feature_cols = [c for c in df.columns if c not in COLS_EXCLUIR]
    print(f"[Tuning] {len(df):,} filas · {len(feature_cols)} features")
    return df, feature_cols


def split_test(df):
    """Test set = año 2025 completo (igual que el pipeline)."""
    return df[df["fecha"].dt.year == 2025].copy()


# ---------------------------------------------------------------------------
# Evaluación en test (2025) con festivos
# ---------------------------------------------------------------------------
def evaluar_test(modelo, df_test, feature_cols, nombre):
    """Devuelve MAE global y MAE en festivos del año 2025."""
    try:
        import holidays as hlib
        festivos_col  = hlib.Colombia(years=[2025])
        fechas_fest   = set(festivos_col.keys())
        df_test       = df_test.copy()
        df_test["es_fest"] = df_test["fecha"].dt.date.isin(fechas_fest)
    except ImportError:
        df_test["es_fest"] = False

    X_te, y_te = preparar_X_y(df_test, feature_cols)
    pred_log   = modelo.predict(X_te)
    pred       = np.maximum(log_inverse(pred_log), 0)

    mae_global  = mean_absolute_error(y_te, pred)

    df_fest = df_test[df_test["es_fest"]]
    if len(df_fest) > 0:
        X_f, y_f = preparar_X_y(df_fest, feature_cols)
        pred_f   = np.maximum(log_inverse(modelo.predict(X_f)), 0)
        mae_fest = mean_absolute_error(y_f, pred_f)
    else:
        mae_fest = float("nan")

    return round(mae_global, 2), round(mae_fest, 2)


# ---------------------------------------------------------------------------
# Entrenamiento con defaults (baseline de comparación)
# ---------------------------------------------------------------------------
def entrenar_con_defaults(df, feature_cols, nombre):
    """Entrena con los hiperparámetros actuales del pipeline."""
    mask_train = df["fecha"].dt.year < 2025
    mask_val   = (df["fecha"] >= "2024-11-01") & (df["fecha"].dt.year < 2025)
    df_train   = df[mask_train & ~mask_val]
    df_val     = df[mask_val]
    df_test    = split_test(df)

    X_tr,  y_tr  = preparar_X_y(df_train, feature_cols)
    X_val, y_val = preparar_X_y(df_val,   feature_cols)
    y_tr  = log_transform(y_tr)
    y_val = log_transform(y_val)

    cat_valid = [c for c in CAT_FEATURES if c in feature_cols]

    t0 = time.time()
    if nombre == "LightGBM":
        m = lgb.LGBMRegressor(**LGBM_DEFAULTS)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(-1)],
              categorical_feature=cat_valid)
    else:
        m = xgb.XGBRegressor(**XGB_DEFAULTS, early_stopping_rounds=100)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    elapsed = time.time() - t0
    mae_g, mae_f = evaluar_test(m, df_test, feature_cols, nombre)
    return mae_g, mae_f, round(elapsed, 1)


# ---------------------------------------------------------------------------
# Objetivo Optuna — expanding window sobre Fold3 + Fold4
# ---------------------------------------------------------------------------
def make_objective_lgbm(df, feature_cols):
    cat_valid = [c for c in CAT_FEATURES if c in feature_cols]

    def objective(trial):
        params = dict(
            num_leaves        = trial.suggest_int("num_leaves", 20, 150),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            n_estimators      = trial.suggest_int("n_estimators", 100, 500),
            min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = 0.8,
            reg_alpha         = 0.1,
            reg_lambda        = 0.1,
            random_state      = 42,
            n_jobs            = -1,
            verbose           = -1,
        )
        maes = []
        for _, tr_ini, tr_fin, te_ini, te_fin in FOLDS_OPTUNA:
            df_tr_full = df[(df["fecha"] >= tr_ini) & (df["fecha"] <= tr_fin)]
            df_te      = df[(df["fecha"] >= te_ini) & (df["fecha"] <= te_fin)]
            if len(df_tr_full) < 200 or len(df_te) < 50:
                continue

            cut    = int(len(df_tr_full) * 0.80)
            df_tr  = df_tr_full.iloc[:cut]
            df_val = df_tr_full.iloc[cut:]

            X_tr,  y_tr  = preparar_X_y(df_tr,  feature_cols)
            X_val, y_val = preparar_X_y(df_val, feature_cols)
            X_te,  y_te  = preparar_X_y(df_te,  feature_cols)
            y_tr  = log_transform(y_tr)
            y_val = log_transform(y_val)

            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)],
                  categorical_feature=cat_valid)

            pred = np.maximum(log_inverse(m.predict(X_te)), 0)
            maes.append(mean_absolute_error(y_te, pred))

        return float(np.mean(maes)) if maes else float("inf")

    return objective


def make_objective_xgb(df, feature_cols):
    def objective(trial):
        params = dict(
            max_depth        = trial.suggest_int("max_depth", 3, 10),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            n_estimators     = trial.suggest_int("n_estimators", 100, 500),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = 10,
            random_state     = 42,
            n_jobs           = -1,
            eval_metric      = "mae",
            verbosity        = 0,
            early_stopping_rounds = 50,
        )
        maes = []
        for _, tr_ini, tr_fin, te_ini, te_fin in FOLDS_OPTUNA:
            df_tr_full = df[(df["fecha"] >= tr_ini) & (df["fecha"] <= tr_fin)]
            df_te      = df[(df["fecha"] >= te_ini) & (df["fecha"] <= te_fin)]
            if len(df_tr_full) < 200 or len(df_te) < 50:
                continue

            cut    = int(len(df_tr_full) * 0.80)
            df_tr  = df_tr_full.iloc[:cut]
            df_val = df_tr_full.iloc[cut:]

            X_tr,  y_tr  = preparar_X_y(df_tr,  feature_cols)
            X_val, y_val = preparar_X_y(df_val, feature_cols)
            X_te,  y_te  = preparar_X_y(df_te,  feature_cols)
            y_tr  = log_transform(y_tr)
            y_val = log_transform(y_val)

            m = xgb.XGBRegressor(**params)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            pred = np.maximum(log_inverse(m.predict(X_te)), 0)
            maes.append(mean_absolute_error(y_te, pred))

        return float(np.mean(maes)) if maes else float("inf")

    return objective


# ---------------------------------------------------------------------------
# Entrenamiento final con mejores params (sobre train completo → evaluar en 2025)
# ---------------------------------------------------------------------------
def entrenar_tuneado(nombre, best_params, df, feature_cols):
    mask_train = df["fecha"].dt.year < 2025
    mask_val   = (df["fecha"] >= "2024-11-01") & (df["fecha"].dt.year < 2025)
    df_train   = df[mask_train & ~mask_val]
    df_val     = df[mask_val]
    df_test    = split_test(df)

    X_tr,  y_tr  = preparar_X_y(df_train, feature_cols)
    X_val, y_val = preparar_X_y(df_val,   feature_cols)
    y_tr  = log_transform(y_tr)
    y_val = log_transform(y_val)

    cat_valid = [c for c in CAT_FEATURES if c in feature_cols]

    t0 = time.time()
    if nombre == "LightGBM":
        params = {**LGBM_DEFAULTS, **best_params,
                  "n_estimators": best_params.get("n_estimators", 500)}
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(-1)],
              categorical_feature=cat_valid)
    else:
        params = {**XGB_DEFAULTS, **best_params,
                  "n_estimators": best_params.get("n_estimators", 500),
                  "early_stopping_rounds": 100}
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    elapsed  = time.time() - t0
    mae_g, mae_f = evaluar_test(m, df_test, feature_cols, nombre)
    return mae_g, mae_f, round(elapsed, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not INPUT.exists():
        print(f"[ERROR] No se encontró {INPUT}")
        print("  Genera el archivo primero con: python run_pipeline.py --steps etl features")
        sys.exit(1)

    df, feature_cols = cargar_datos()
    resultados = {}

    # -----------------------------------------------------------------------
    for nombre, make_obj in [("LightGBM", make_objective_lgbm),
                              ("XGBoost",  make_objective_xgb)]:
        print(f"\n{'='*60}")
        print(f"  TUNING {nombre} — 50 trials (objetivo: MAE val Fold3+Fold4)")
        print(f"{'='*60}")

        # — Baseline: defaults del pipeline —
        print(f"  Entrenando {nombre} con defaults actuales...")
        mae_def_g, mae_def_f, t_def = entrenar_con_defaults(df, feature_cols, nombre)
        print(f"  MAE defaults (global)   : {mae_def_g:,.2f}")
        print(f"  MAE defaults (festivos) : {mae_def_f:,.2f}")

        # — Optuna —
        print(f"\n  Iniciando 50 trials de Optuna...")
        t_optuna_ini = time.time()
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_obj(df, feature_cols), n_trials=50,
                       show_progress_bar=True)
        t_optuna = round((time.time() - t_optuna_ini) / 60, 1)

        best_params = study.best_params
        best_val    = round(study.best_value, 2)
        print(f"\n  Mejor MAE en validación : {best_val:,.2f}")
        print(f"  Tiempo de tuning        : {t_optuna} min")
        print(f"  Mejores parámetros      : {best_params}")

        # — Entrenamiento final con mejores params → test 2025 —
        print(f"\n  Entrenando {nombre} tuneado sobre train completo...")
        mae_tun_g, mae_tun_f, t_tun = entrenar_tuneado(nombre, best_params, df, feature_cols)

        mejora_g = (mae_def_g - mae_tun_g) / mae_def_g * 100
        mejora_f = (mae_def_f - mae_tun_f) / mae_def_f * 100 if mae_def_f == mae_def_f else float("nan")

        resultados[nombre] = {
            "mae_defaults_global":   mae_def_g,
            "mae_defaults_festivos": mae_def_f,
            "mae_tuneado_global":    mae_tun_g,
            "mae_tuneado_festivos":  mae_tun_f,
            "mejora_global_pct":     round(mejora_g, 2),
            "mejora_festivos_pct":   round(mejora_f, 2),
            "t_tuning_min":          t_optuna,
            "t_train_final_s":       t_tun,
            "best_params":           best_params,
            "best_val_mae":          best_val,
        }

    # -----------------------------------------------------------------------
    # Reporte final
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("=== RESULTADO DEL TUNING ===")
    print("=" * 60)

    for nombre, r in resultados.items():
        mejora_g = r["mejora_global_pct"]
        signo    = "(mejora)" if mejora_g > 0 else "(empeoro)"
        params_str = str(r['best_params'])
        print(nombre + ":")
        print(f"  MAE actual (defaults)  - global  : {r['mae_defaults_global']:>10,.2f} pax")
        print(f"  MAE actual (defaults)  - festivos: {r['mae_defaults_festivos']:>10,.2f} pax")
        print(f"  MAE tuneado            - global  : {r['mae_tuneado_global']:>10,.2f} pax")
        print(f"  MAE tuneado            - festivos: {r['mae_tuneado_festivos']:>10,.2f} pax")
        print(f"  Mejora global          : {mejora_g:>+.2f}% {signo}")
        print(f"  Mejora en festivos     : {r['mejora_festivos_pct']:>+.2f}%")
        print(f"  Tiempo tuning          : {r['t_tuning_min']} min")
        print(f"  Tiempo train final     : {r['t_train_final_s']} s")
        print(f"  Mejor MAE val (Optuna) : {r['best_val_mae']:,.2f}")
        print(f"  Mejores hiperparametros: {params_str}")
        print()

    print("\n" + "-" * 60)
    print("Vale la pena implementar?")
    for nombre, r in resultados.items():
        mejora = r["mejora_global_pct"]
        fn = nombre.lower().replace("boost", "b").replace("lightgbm", "lgbm")
        if mejora > 5:
            print(f"  {nombre}: SI ({mejora:.2f}% > 5%) - RECOMENDADO implementar en 03_models.py")
            print(f"    Reemplazar hiperparametros en entrenar_{fn}()")
            print(f"    Parametros: {r['best_params']}")
        elif mejora > 0:
            print(f"  {nombre}: MARGINAL ({mejora:.2f}%, < 5%) - complejidad extra no justifica la ganancia")
        else:
            print(f"  {nombre}: NO ({mejora:.2f}%) - defaults actuales son mejores")
    print("=" * 60)


if __name__ == "__main__":
    main()
