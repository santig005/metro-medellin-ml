"""
AWS Upload — Metro de Medellín
Sube los artefactos del pipeline a S3 (metro-medellin-datalake).
Solo actúa si METRO_MODE=aws. En modo local imprime resumen y termina limpiamente.
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BUCKET = "metro-medellin-datalake"

ARTEFACTOS = [
    ("data/processed/trusted.parquet",       "trusted/trusted.parquet",                             {}),
    ("data/processed/featured.parquet",      "refined/model/features/featured.parquet",             {}),
    ("data/output/predictions.csv",          "refined/model/predictions/predictions.csv",            {}),
    ("data/output/metrics.json",             "refined/model/metrics/metrics.json",                   {}),
    ("data/output/feature_importance.csv",   "refined/model/metrics/feature_importance.csv",         {}),
    ("data/output/metro_dashboard.html",     "app/metro_dashboard.html",
     {"ContentType": "text/html"}),
    ("data/output/model_lgbm.pkl",           "models/model_lgbm.pkl",                               {}),
]


def subir_archivo(s3, local_path: str, s3_key: str, extra_args: dict) -> bool:
    if not os.path.exists(local_path):
        print(f"  [SKIP] No encontrado localmente: {local_path}")
        return False
    try:
        kwargs = {"ExtraArgs": extra_args} if extra_args else {}
        s3.upload_file(local_path, BUCKET, s3_key, **kwargs)
        size_kb = os.path.getsize(local_path) / 1024
        print(f"  [OK]   {local_path:<45} → s3://{BUCKET}/{s3_key}  ({size_kb:,.0f} KB)")
        return True
    except Exception as e:
        print(f"  [ERR]  {local_path}: {e}")
        return False


def main():
    mode = os.getenv("METRO_MODE", "local")

    if mode != "aws":
        print("[AWS Upload] METRO_MODE != 'aws' — upload omitido (modo local).")
        print(f"[AWS Upload] Para activar: set METRO_MODE=aws y ejecutar de nuevo.")
        return

    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
    except ImportError:
        print("[AWS Upload] boto3 no está instalado. Ejecuta: pip install boto3")
        sys.exit(1)

    print(f"[AWS Upload] Iniciando subida a s3://{BUCKET}/ ...")

    try:
        s3 = boto3.client("s3")
        # Verificar acceso al bucket antes de intentar subir
        s3.head_bucket(Bucket=BUCKET)
    except Exception as e:
        if "NoCredentials" in type(e).__name__ or "NoCredentialsError" in str(type(e)):
            print("[AWS Upload] Error: credenciales AWS no configuradas.")
            print("  Configura con: aws configure  (o variables AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)")
        elif "404" in str(e) or "NoSuchBucket" in str(e):
            print(f"[AWS Upload] Error: el bucket '{BUCKET}' no existe o no tienes acceso.")
        else:
            print(f"[AWS Upload] Error al conectar con S3: {e}")
        sys.exit(1)

    subidos = 0
    for local_path, s3_key, extra in ARTEFACTOS:
        if subir_archivo(s3, local_path, s3_key, extra):
            subidos += 1

    print(f"\n[AWS Upload] Completado: {subidos}/{len(ARTEFACTOS)} archivos subidos a s3://{BUCKET}/")


if __name__ == "__main__":
    main()
