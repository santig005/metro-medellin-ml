"""
Orquestador del pipeline completo.
Ejecuta los 4 scripts en orden desde el directorio raíz del proyecto.
Uso: python run_pipeline.py
"""

import os
import sys
import subprocess
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASOS = [
    ("01_etl.py",        "ETL — Carga, limpieza y parseo de fechas"),
    ("02_features.py",   "Feature Engineering — Lags, rolling, festivos, encoding"),
    ("03_models.py",     "Modelos — Baseline, RF, LightGBM, XGBoost"),
    ("04_dashboard.py",  "Dashboard — HTML interactivo con Plotly"),
    ("05_aws_upload.py", "AWS Upload — Subida de artefactos a S3 (solo si METRO_MODE=aws)"),
]


def ejecutar_paso(script: str, descripcion: str) -> bool:
    separador = "=" * 60
    print(f"\n{separador}")
    print(f"  {descripcion}")
    print(f"  Script: src/{script}")
    print(separador)

    inicio = time.time()
    result = subprocess.run(
        [sys.executable, "-X", "utf8", f"src/{script}"],
        capture_output=False,
    )
    duracion = time.time() - inicio

    if result.returncode != 0:
        print(f"\n[ERROR] {script} falló (código {result.returncode}) en {duracion:.1f}s")
        return False

    print(f"\n[OK] {script} completado en {duracion:.1f}s")
    return True


def main():
    # Moverse al directorio del script para rutas relativas consistentes
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("  METRO DE MEDELLÍN — Pipeline de ML")
    print("  Universidad EAFIT · Equipo Git Hug")
    print("=" * 60)

    inicio_total = time.time()
    for script, desc in PASOS:
        ok = ejecutar_paso(script, desc)
        if not ok:
            print(f"\n[ABORTADO] El pipeline se detuvo en {script}.")
            sys.exit(1)

    total = time.time() - inicio_total
    print("\n" + "=" * 60)
    print(f"  Pipeline completado en {total:.1f}s")
    print("  Outputs:")
    print("    data/processed/trusted.parquet")
    print("    data/processed/featured.parquet")
    print("    data/output/predictions.csv")
    print("    data/output/metrics.json")
    print("    data/output/feature_importance.csv")
    print("    data/output/model_lgbm.pkl")
    print("    data/output/metro_dashboard.html")
    print("    data/output/eda_notes.md")
    print("    [S3: solo si METRO_MODE=aws]")
    print("=" * 60)


if __name__ == "__main__":
    main()
