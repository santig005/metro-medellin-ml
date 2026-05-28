"""
ETL — Metro de Medellín
Carga, parseo de fechas mixto, limpieza, unpivot wide→long,
asignación de tipo_linea y validaciones de calidad.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

# Forzar UTF-8 en la consola de Windows para que los caracteres especiales se muestren bien
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN — permite cambiar de fuente sin tocar el resto del código
# ---------------------------------------------------------------------------
MODE = os.getenv("METRO_MODE", "local")

# Rutas (relativas al directorio raíz del proyecto)
RAW_LOCAL  = "data/raw/afluencia-metro.csv"
S3_PATH    = os.getenv("METRO_S3_PATH", "s3://metro-medellin-data/raw/afluencia-metro.csv")
OUTPUT     = "data/processed/trusted.parquet"

# Columnas de horas presentes en el CSV
HORAS = [f"{h}:00" for h in range(4, 24)]   # 4:00 … 23:00

# Mapeo de líneas a tipo_linea
TIPO_LINEA = {
    "LÍNEA A": "metro_ferreo",
    "LÍNEA B": "metro_ferreo",
    "LÍNEA H": "metrocable",
    "LÍNEA J": "metrocable",
    "LÍNEA K": "metrocable",
    "LÍNEA L": "metrocable",
    "LÍNEA M": "metrocable",
    "LÍNEA P": "metrocable",
    "LÍNEA T-A": "tranvia",
    "LÍNEA 1": "brt",
    "LÍNEA 2": "brt",
    "LÍNEA O": "brt",
}


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------
def cargar_csv() -> pd.DataFrame:
    if MODE == "aws":
        print(f"[ETL] Modo AWS — leyendo desde {S3_PATH}")
        # Lee desde S3 con pyarrow; requiere boto3 y credenciales AWS configuradas
        import s3fs
        fs = s3fs.S3FileSystem()
        with fs.open(S3_PATH) as f:
            df = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
    else:
        print(f"[ETL] Modo local — leyendo desde {RAW_LOCAL}")
        df = pd.read_csv(RAW_LOCAL, encoding="utf-8-sig", dtype=str)

    print(f"[ETL] Filas cargadas: {len(df):,}  |  Columnas: {len(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# 2. PARSEO DE FECHAS (formato mixto D/M/YYYY en 2023-24, M/D/YYYY en 2025)
# ---------------------------------------------------------------------------
def parsear_fechas(df: pd.DataFrame) -> pd.DataFrame:
    raw = df["DIA"].copy()

    # Primera pasada: dayfirst=True (formato 2023-24)
    fechas = pd.to_datetime(raw, dayfirst=True, errors="coerce")
    nulos_primera = fechas.isna().sum()

    # Segunda pasada para los NaN: dayfirst=False (formato 2025)
    mascara_nulos = fechas.isna()
    fechas[mascara_nulos] = pd.to_datetime(
        raw[mascara_nulos], dayfirst=False, errors="coerce"
    )
    nulos_segunda = fechas.isna().sum()

    print(f"[ETL] Fechas: {nulos_primera} NaN tras primera pasada, "
          f"{nulos_segunda} NaN tras segunda pasada")

    if nulos_segunda > 0:
        print(f"  ⚠  Fechas que no se pudieron parsear:\n  {raw[fechas.isna()].unique()}")

    df["fecha"] = fechas
    df["DIA_raw"] = raw   # conservar el original para auditoría
    return df


# ---------------------------------------------------------------------------
# 3. LIMPIEZA NUMÉRICA
# ---------------------------------------------------------------------------
def limpiar_numericos(df: pd.DataFrame) -> pd.DataFrame:
    cols_num = HORAS + ["TOTAL"]
    for col in cols_num:
        if col in df.columns:
            # Remover comas de miles y convertir a float
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"nan": np.nan, "": np.nan, "None": np.nan})
                .astype(float)
            )
    return df


# ---------------------------------------------------------------------------
# 4. UNPIVOT wide → long
# ---------------------------------------------------------------------------
def unpivot(df: pd.DataFrame) -> pd.DataFrame:
    id_vars = ["fecha", "DIA_raw", "LINEA", "TOTAL"]
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=HORAS,
        var_name="hora_str",
        value_name="pasajeros",
    )
    # Convertir "4:00" → 4 (entero)
    df_long["hora"] = df_long["hora_str"].str.replace(":00", "", regex=False).astype(int)
    df_long = df_long.drop(columns=["hora_str"])

    # Renombrar LINEA → linea para consistencia
    df_long = df_long.rename(columns={"LINEA": "linea"})

    print(f"[ETL] Registros tras unpivot: {len(df_long):,}")
    return df_long


# ---------------------------------------------------------------------------
# 5. TIPO DE LÍNEA
# ---------------------------------------------------------------------------
def agregar_tipo_linea(df: pd.DataFrame) -> pd.DataFrame:
    df["tipo_linea"] = df["linea"].map(TIPO_LINEA)
    sin_tipo = df[df["tipo_linea"].isna()]["linea"].unique()
    if len(sin_tipo) > 0:
        print(f"  ⚠  Líneas sin tipo asignado: {sin_tipo}")
    return df


# ---------------------------------------------------------------------------
# 6. VALIDACIONES DE CALIDAD
# ---------------------------------------------------------------------------
def validar_calidad(df_long: pd.DataFrame, df_wide: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("VALIDACIONES DE CALIDAD")
    print("=" * 60)

    # 6.1 Rango de fechas por año
    print("\n--- Rango de fechas por año ---")
    for yr in [2023, 2024, 2025]:
        sub = df_long[df_long["fecha"].dt.year == yr]["fecha"]
        if len(sub) > 0:
            dias_unicos = sub.dt.date.nunique()
            print(f"  {yr}: {sub.min().date()} → {sub.max().date()}  "
                  f"({dias_unicos} días únicos)")
        else:
            print(f"  {yr}: SIN DATOS ⚠")

    # 6.2 Pasajeros negativos y outliers extremos
    print("\n--- Outliers de pasajeros ---")
    pax = df_long["pasajeros"].dropna()
    negativos = (pax < 0).sum()
    p999 = pax.quantile(0.999)
    extremos = (pax > p999).sum()
    print(f"  Negativos: {negativos}")
    print(f"  Percentil 99.9: {p999:,.0f} pasajeros")
    print(f"  Registros > p99.9: {extremos} ({extremos/len(pax)*100:.3f}%)")
    if extremos > 0:
        top_outliers = df_long[df_long["pasajeros"] > p999][
            ["fecha", "linea", "hora", "pasajeros"]
        ].nlargest(5, "pasajeros")
        print("  Top 5 outliers:")
        print(top_outliers.to_string(index=False))

    # 6.3 Líneas con más del 20% de horas en NaN
    print("\n--- Cobertura de horas por línea ---")
    total_horas = len(HORAS)
    for linea, grp in df_long.groupby("linea"):
        tasa_nan = grp["pasajeros"].isna().mean()
        flag = " ⚠  >20% NaN" if tasa_nan > 0.2 else ""
        print(f"  {linea}: {tasa_nan*100:.1f}% NaN{flag}")

    # 6.4 Verificar que TOTAL == suma de horas
    print("\n--- Consistencia TOTAL vs suma de horas ---")
    suma_horas = (
        df_long.groupby(["fecha", "linea"])["pasajeros"]
        .sum(min_count=1)
        .reset_index(name="suma_calculada")
    )
    # Unir con TOTAL del wide
    totales_wide = df_wide[["fecha", "LINEA", "TOTAL"]].rename(
        columns={"LINEA": "linea"}
    )
    comp = suma_horas.merge(totales_wide, on=["fecha", "linea"], how="left")
    comp = comp.dropna(subset=["suma_calculada", "TOTAL"])
    comp["diff"] = (comp["suma_calculada"] - comp["TOTAL"]).abs()
    discrepancias = comp[comp["diff"] > 10]  # tolerancia de 10 pasajeros por redondeos
    print(f"  Comparaciones realizadas: {len(comp):,}")
    print(f"  Discrepancias (diff > 10): {len(discrepancias):,}")
    if len(discrepancias) > 0:
        print(f"  Diff máxima: {comp['diff'].max():,.0f}")
        print("  Muestra de discrepancias:")
        print(discrepancias.head(3)[["fecha","linea","suma_calculada","TOTAL","diff"]].to_string(index=False))

    # 6.5 Líneas que aparecen/desaparecen entre años
    print("\n--- Presencia de líneas por año ---")
    for linea in sorted(df_long["linea"].unique()):
        años_presentes = sorted(
            df_long[df_long["linea"] == linea]["fecha"].dt.year.unique()
        )
        print(f"  {linea}: años {años_presentes}")

    # 6.6 Distribución pasajeros=0 vs NaN
    print("\n--- Pasajeros=0 vs NaN (¿son lo mismo operacionalmente?) ---")
    ceros = (df_long["pasajeros"] == 0).sum()
    nans  = df_long["pasajeros"].isna().sum()
    total = len(df_long)
    print(f"  Pasajeros = 0  : {ceros:,} ({ceros/total*100:.2f}%)")
    print(f"  Pasajeros = NaN: {nans:,} ({nans/total*100:.2f}%)")
    print(f"  → NaN en horas nocturnas son estructurales (línea no opera esa hora)")
    # Desglose NaN por línea y hora
    nan_por_hora = df_long[df_long["pasajeros"].isna()].groupby("hora").size()
    print(f"  Horas con más NaN: {nan_por_hora.nlargest(5).to_dict()}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 7. ESTADÍSTICAS FINALES
# ---------------------------------------------------------------------------
def resumen_estadistico(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RESUMEN ESTADÍSTICO FINAL")
    print("=" * 60)
    print(f"  Total registros: {len(df):,}")
    print(f"  Rango de fechas: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"  Líneas: {sorted(df['linea'].unique())}")
    print(f"  Horas: {sorted(df['hora'].unique())}")
    print()
    print("  Pasajeros por tipo_linea:")
    resumen = df.groupby("tipo_linea")["pasajeros"].agg(
        media="mean", mediana="median", max="max", total="sum"
    ).round(1)
    print(resumen.to_string())
    print()
    print("  Pasajeros por línea (media/hora):")
    por_linea = df.groupby("linea")["pasajeros"].agg(
        registros="count", media="mean", max="max"
    ).round(1).sort_values("media", ascending=False)
    print(por_linea.to_string())
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)

    # Paso 1: Carga
    df_wide = cargar_csv()

    # Paso 2: Parseo de fechas
    df_wide = parsear_fechas(df_wide)

    # Paso 3: Limpieza numérica
    df_wide = limpiar_numericos(df_wide)

    # Paso 4: Unpivot
    df_long = unpivot(df_wide)

    # Paso 5: Tipo de línea
    df_long = agregar_tipo_linea(df_long)

    # Paso 6: Validaciones
    validar_calidad(df_long, df_wide.rename(columns={"DIA": "DIA_orig"}))

    # Guardar: excluimos la columna TOTAL del long (no es una feature, es derivada)
    # y mantenemos NaN en horas donde la línea no opera
    columnas_finales = ["fecha", "DIA_raw", "linea", "tipo_linea", "hora", "pasajeros", "TOTAL"]
    # TOTAL en df_long viene del wide y representa el total diario por línea
    df_output = df_long[columnas_finales].copy()

    df_output = df_output.sort_values(["fecha", "linea", "hora"]).reset_index(drop=True)
    df_output.to_parquet(OUTPUT, index=False)
    print(f"[ETL] Guardado: {OUTPUT}  ({len(df_output):,} registros)")

    # Paso 7: Resumen
    resumen_estadistico(df_output)


if __name__ == "__main__":
    main()
