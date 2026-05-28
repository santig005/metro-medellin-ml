"""
Feature Engineering — Metro de Medellín
Lee trusted.parquet y construye todas las features sin data leakage.
Guarda featured.parquet y los encoders en JSON.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import holidays

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

INPUT  = "data/processed/trusted.parquet"
OUTPUT = "data/processed/featured.parquet"
ENCODER_PATH = "data/processed/label_encoders.json"


# ---------------------------------------------------------------------------
# 1. FEATURES TEMPORALES BÁSICAS
# ---------------------------------------------------------------------------
def features_temporales(df: pd.DataFrame) -> pd.DataFrame:
    df["hora_del_dia"]  = df["hora"].astype(int)
    df["dia_semana"]    = df["fecha"].dt.dayofweek          # 0=lunes
    df["mes"]           = df["fecha"].dt.month
    df["año"]           = df["fecha"].dt.year
    df["dia_del_año"]   = df["fecha"].dt.dayofyear
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["semana_del_año"] = df["fecha"].dt.isocalendar().week.astype(int)
    # Tendencia interanual: 0=2023, 1=2024, 2=2025
    # El sistema muestra declive de ~2.9% (2023→2024) y ~8.4% (2024→2025).
    # Permite al modelo capturar la tendencia de caída en demanda.
    df["año_trend"] = df["año"] - 2023
    return df


# ---------------------------------------------------------------------------
# 2. FESTIVOS COLOMBIANOS Y PUENTES
# ---------------------------------------------------------------------------
def features_festivos(df: pd.DataFrame) -> pd.DataFrame:
    festivos_col = holidays.Colombia(years=[2023, 2024, 2025])
    fechas_festivas = set(festivos_col.keys())

    df["es_festivo"] = df["fecha"].dt.date.apply(lambda d: d in fechas_festivas).astype(int)

    # Puente: lunes siguiente a un festivo en fin de semana (Colombia usa la ley de puentes)
    # También se marca el lunes cuando el festivo es trasladado
    fechas_puente = set()
    for f in fechas_festivas:
        import datetime
        d = pd.Timestamp(f)
        # Si el festivo es martes-domingo, el lunes anterior podría ser puente
        # En Colombia la ley de puentes traslada algunos festivos al lunes siguiente
        # holidays.Colombia ya aplica esa ley — los festivos que caen en lunes son los puentes
        if d.dayofweek == 0:  # ya es lunes → es el puente
            fechas_puente.add(d.date())

    df["es_puente"] = df["fecha"].dt.date.apply(lambda d: d in fechas_puente).astype(int)
    return df


# ---------------------------------------------------------------------------
# 3. VARIABLES CÍCLICAS (evitan discontinuidad en modelos lineales)
# ---------------------------------------------------------------------------
def features_ciclicas(df: pd.DataFrame) -> pd.DataFrame:
    df["hora_sin"]         = np.sin(2 * np.pi * df["hora_del_dia"] / 24)
    df["hora_cos"]         = np.cos(2 * np.pi * df["hora_del_dia"] / 24)
    df["dia_semana_sin"]   = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_semana_cos"]   = np.cos(2 * np.pi * df["dia_semana"] / 7)
    df["mes_sin"]          = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"]          = np.cos(2 * np.pi * df["mes"] / 12)
    return df


# ---------------------------------------------------------------------------
# 4. ENCODING DE LÍNEA
# ---------------------------------------------------------------------------
def features_encoding(df: pd.DataFrame) -> dict:
    """Retorna df con columnas encoded y el dict de mapeos."""
    # Label encoding manual para tener control del mapeo
    lineas      = sorted(df["linea"].unique())
    tipos_linea = sorted(df["tipo_linea"].unique())

    linea_map      = {v: i for i, v in enumerate(lineas)}
    tipo_linea_map = {v: i for i, v in enumerate(tipos_linea)}

    df["linea_encoded"]      = df["linea"].map(linea_map)
    df["tipo_linea_encoded"] = df["tipo_linea"].map(tipo_linea_map)

    # Flag especial: metro férreo tiene patrones muy distintos
    df["linea_es_metro_ferreo"] = df["tipo_linea"].isin(["metro_ferreo"]).astype(int)

    encoders = {
        "linea": linea_map,
        "tipo_linea": tipo_linea_map,
        "linea_inv": {v: k for k, v in linea_map.items()},
        "tipo_linea_inv": {v: k for k, v in tipo_linea_map.items()},
    }
    return df, encoders


# ---------------------------------------------------------------------------
# 5. LAGS — calculados por grupo (linea, hora_del_dia) para no mezclar series
# ---------------------------------------------------------------------------
def features_lags(df: pd.DataFrame) -> pd.DataFrame:
    # Ordenar para garantizar que los lags sean correctos
    df = df.sort_values(["linea", "hora_del_dia", "fecha"]).reset_index(drop=True)

    def calc_lags(grp):
        # Crear índice de días únicos para hacer shift limpio
        grp = grp.sort_values("fecha").copy()
        grp["lag_1d"]  = grp["pasajeros"].shift(1)   # 1 día atrás (misma línea+hora)
        grp["lag_7d"]  = grp["pasajeros"].shift(7)   # 7 días atrás
        grp["lag_14d"] = grp["pasajeros"].shift(14)  # 14 días atrás
        return grp

    print("[Features] Calculando lags por (linea, hora)...")
    df = df.groupby(["linea", "hora_del_dia"], group_keys=False).apply(calc_lags)
    return df


# ---------------------------------------------------------------------------
# 6. ROLLING MEANS — ventanas sobre días anteriores (sin incluir el día actual)
# ---------------------------------------------------------------------------
def features_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["linea", "hora_del_dia", "fecha"]).reset_index(drop=True)

    def calc_rolling(grp):
        grp = grp.sort_values("fecha").copy()
        pax = grp["pasajeros"]
        # min_periods=1 para no perder demasiados registros al inicio
        grp["rolling_7d"]     = pax.shift(1).rolling(7,  min_periods=3).mean()
        grp["rolling_28d"]    = pax.shift(1).rolling(28, min_periods=14).mean()
        grp["rolling_7d_std"] = pax.shift(1).rolling(7,  min_periods=3).std()
        return grp

    print("[Features] Calculando rolling means por (linea, hora)...")
    df = df.groupby(["linea", "hora_del_dia"], group_keys=False).apply(calc_rolling)
    return df


# ---------------------------------------------------------------------------
# 7. INTERACCIONES DE ALTO VALOR
# ---------------------------------------------------------------------------
def features_interacciones(df: pd.DataFrame) -> pd.DataFrame:
    df["hora_x_dia_semana"]   = df["hora_del_dia"] * df["dia_semana"]
    df["es_hora_pico_manana"] = df["hora_del_dia"].between(6, 9).astype(int)
    df["es_hora_pico_tarde"]  = df["hora_del_dia"].between(17, 19).astype(int)
    return df


# ---------------------------------------------------------------------------
# 8. REPORTE DE CORRELACIONES Y NaN POR LAGS
# ---------------------------------------------------------------------------
def reporte_features(df: pd.DataFrame, df_antes: int) -> None:
    print("\n" + "=" * 60)
    print("REPORTE DE FEATURE ENGINEERING")
    print("=" * 60)

    total_antes = df_antes
    nulos_lag = df[["lag_1d", "lag_7d", "lag_14d", "rolling_7d", "rolling_28d"]].isna().any(axis=1).sum()
    print(f"\n  Registros antes de lags/rolling: {total_antes:,}")
    print(f"  Registros con NaN en lags o rolling: {nulos_lag:,}")
    print(f"  Registros descartables: {nulos_lag:,} "
          f"({nulos_lag/total_antes*100:.1f}%) — solo se usan en train/test")

    # Correlaciones con pasajeros (solo registros no NaN)
    df_valido = df.dropna(subset=["pasajeros"])
    features_num = [
        "hora_del_dia", "dia_semana", "mes", "año", "año_trend", "dia_del_año",
        "es_fin_de_semana", "es_festivo", "es_puente", "hora_sin", "hora_cos",
        "dia_semana_sin", "dia_semana_cos", "mes_sin", "mes_cos",
        "linea_encoded", "tipo_linea_encoded", "linea_es_metro_ferreo",
        "lag_1d", "lag_7d", "lag_14d",
        "rolling_7d", "rolling_28d", "rolling_7d_std",
        "hora_x_dia_semana", "es_hora_pico_manana", "es_hora_pico_tarde",
    ]
    features_disponibles = [f for f in features_num if f in df_valido.columns]

    corrs = df_valido[features_disponibles + ["pasajeros"]].corr()["pasajeros"].drop("pasajeros")
    corrs_sorted = corrs.abs().sort_values(ascending=False)

    print("\n  Top 15 features por correlación con pasajeros (|r|):")
    for feat, val in corrs_sorted.head(15).items():
        signo = "+" if corrs[feat] >= 0 else "-"
        print(f"    {signo}{val:.3f}  {feat}")

    print()
    print(f"  Total features generadas: {len(features_disponibles)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("[Features] Leyendo trusted.parquet...")
    df = pd.read_parquet(INPUT)
    total_antes = len(df)
    print(f"[Features] Registros de entrada: {total_antes:,}")

    # Features temporales primero (crea hora_del_dia y otras columnas base)
    print("[Features] Features temporales...")
    df = features_temporales(df)

    # --- ORDEN CRÍTICO: lags y rolling se calculan sobre pasajeros ORIGINALES ---
    # Si primero llenamos NaN con 0, el lag_1d de una hora operativa recibiría
    # el 0 artificial de la hora no-operativa anterior (p.e. LÍNEA L opera lunes
    # pero no domingo: su lag del lunes tomaría el 0 del domingo → sesgo).
    # Solución: calcular lags ANTES del 0-fill, luego rellenar el target.
    df = features_lags(df)
    df = features_rolling(df)

    # Ahora sí rellenar NaN estructurales con 0 en el target (horas no-operativas).
    # "No-operativa" = nunca tiene datos reales para esa (línea, hora).
    horas_operativas = (
        df[df["pasajeros"].notna()]
        .groupby(["linea", "hora_del_dia"])
        .size()
        .reset_index(name="n")
    )
    horas_operativas["opera"] = True
    df = df.merge(
        horas_operativas[["linea", "hora_del_dia", "opera"]],
        on=["linea", "hora_del_dia"], how="left"
    )
    df.loc[df["opera"].isna() & df["pasajeros"].isna(), "pasajeros"] = 0
    df = df.drop(columns=["opera"])

    print("[Features] Festivos colombianos...")
    df = features_festivos(df)

    print("[Features] Variables cíclicas...")
    df = features_ciclicas(df)

    print("[Features] Encoding de líneas...")
    df, encoders = features_encoding(df)

    # Guardar encoders para predicciones futuras
    os.makedirs("data/processed", exist_ok=True)
    with open(ENCODER_PATH, "w", encoding="utf-8") as f:
        json.dump(encoders, f, ensure_ascii=False, indent=2)
    print(f"[Features] Encoders guardados en {ENCODER_PATH}")

    print("[Features] Interacciones...")
    df = features_interacciones(df)

    # Ordenar cronológicamente para el split temporal posterior
    df = df.sort_values(["fecha", "linea", "hora_del_dia"]).reset_index(drop=True)

    # Reporte
    reporte_features(df, total_antes)

    # Guardar
    df.to_parquet(OUTPUT, index=False)
    print(f"[Features] Guardado: {OUTPUT}  ({len(df):,} registros)")
    print(f"[Features] Columnas: {list(df.columns)}")


if __name__ == "__main__":
    main()
