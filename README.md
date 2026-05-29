# Metro de Medellín — Predicción de Afluencia con ML

**Universidad EAFIT · Proyecto Integrador · Equipo Git Hug · Mayo 2026**

Predicción horaria de pasajeros en las 12 líneas del sistema Metro de Medellín (metro férreo, cables, tranvía y BRT) usando datos de 2023 a 2025. El objetivo central no es "ganar" al baseline en días normales —donde la periodicidad del sistema lo hace casi imbatible— sino demostrar el valor operativo del ML en festivos, donde el baseline falla 3 veces más.

---

## Estructura del proyecto

```
metro-medellin-ml/
├── data/
│   ├── raw/
│   │   └── afluencia-metro.csv          # Fuente original (no modificar)
│   ├── processed/
│   │   ├── trusted.parquet              # Generado por 01_etl.py
│   │   ├── featured.parquet             # Generado por 02_features.py
│   │   └── label_encoders.json          # Mapeo linea/tipo → entero
│   └── output/
│       ├── predictions.csv              # Predicciones de todos los modelos
│       ├── metrics.json                 # Métricas completas (global + festivos + por línea)
│       ├── feature_importance.csv       # Importancia de features — LightGBM
│       ├── model_lgbm.pkl               # Modelo LightGBM global serializado
│       ├── models_per_line.pkl          # Dict con 12 LightGBM por línea
│       ├── metro_dashboard.html         # Dashboard interactivo (~0.96 MB, autocontenido)
│       └── eda_notes.md                 # Hallazgos documentados durante el EDA
├── src/
│   ├── 01_etl.py                        # Carga, limpieza y parseo de fechas
│   ├── 02_features.py                   # Feature engineering completo
│   ├── 03_models.py                     # Entrenamiento y evaluación de modelos
│   ├── 04_dashboard.py                  # Construcción del dashboard en HTML
│   ├── 05_aws_upload.py                 # Subida de artefactos a S3 (solo si METRO_MODE=aws)
│   └── 06_api.py                        # FastAPI para servir predicciones (uvicorn)
├── experiments/
│   └── tune_hyperparams.py              # Optuna tuning LightGBM + XGBoost (no toca pipeline)
├── app.py                               # Dashboard Streamlit (Streamlit Cloud)
├── run_pipeline.py                      # Orquestador: ejecuta los 5 pasos en secuencia
└── requirements.txt
```

---

## Cómo ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo (~130 segundos)
python run_pipeline.py

# O paso a paso
python -X utf8 src/01_etl.py
python -X utf8 src/02_features.py
python -X utf8 src/03_models.py
python -X utf8 src/04_dashboard.py
python -X utf8 src/05_aws_upload.py   # Solo activo si METRO_MODE=aws

# Walk-forward cross-validation (no modifica predictions.csv ni metrics.json)
python -X utf8 src/03_models.py --cv

# Experimento de tuning de hiperparámetros (no modifica el pipeline)
python experiments/tune_hyperparams.py

# API de predicción (FastAPI + uvicorn)
uvicorn src.06_api:app --reload --port 8000

# Dashboard Streamlit (desarrollo local)
streamlit run app.py

# Subida a S3 en entorno AWS EMR
set METRO_MODE=aws
python run_pipeline.py
```

> **Nota Windows:** el flag `-X utf8` es necesario para que los prints con tildes y eñes no fallen en consolas con codificación cp1252.

El output final es `data/output/metro_dashboard.html`. Abrir en cualquier navegador; no requiere servidor.

---

## Flujo del pipeline

### Paso 1 — ETL (`01_etl.py`)

Lee `afluencia-metro.csv` (codificación UTF-8-sig, números con coma de miles) y produce un parquet limpio en formato largo: una fila por `(fecha, línea, hora)`.

Transformaciones principales:
- **Unpivot horario:** el CSV tiene una columna por hora (`4:00`, `5:00`, … `23:00`). Se hace `melt()` para pasar a formato largo.
- **Parseo de fechas mixto** (ver hallazgos): doble pasada con `dayfirst=True` y luego `dayfirst=False` para los registros que quedaron NaN.
- **Tipificación de líneas:** se asigna `tipo_linea` (metro_ferreo, metrocable, tranvia, brt) a cada línea mediante un diccionario estático basado en la nomenclatura oficial del Metro.
- Validación al cierre: 0 fechas no parseadas, 0 pasajeros negativos, 236,220 registros.

### Paso 2 — Feature Engineering (`02_features.py`)

Construye 27 columnas adicionales sin introducir data leakage.

| Grupo | Features |
|-------|----------|
| Temporales básicas | hora_del_dia, dia_semana, mes, año, dia_del_año, semana_del_año, es_fin_de_semana, año_trend |
| Festivos | es_festivo, es_puente (usando `holidays.Colombia`) |
| Cíclicas | hora_sin/cos, dia_semana_sin/cos, mes_sin/cos |
| Encoding | linea_encoded, tipo_linea_encoded, linea_es_metro_ferreo |
| Lags | lag_1d, lag_7d, lag_14d |
| Rolling | rolling_7d, rolling_28d, rolling_7d_std |
| Interacciones | hora_x_dia_semana, es_hora_pico_manana, es_hora_pico_tarde |

**Orden crítico de cálculo (ver complicaciones):** los lags se calculan sobre los valores originales, *antes* de rellenar con 0 las horas no operativas. Si el orden se invirtiera, el lag_1d de un lunes tomaría el 0 artificial del domingo para líneas como LÍNEA L (que solo opera en horario reducido), sesgando todos los modelos.

### Paso 3 — Modelos (`03_models.py`)

Split temporal estricto —sin shuffle, sin validación cruzada que mezcle fechas:

| Conjunto | Período | Registros |
|----------|---------|-----------|
| Train | 2023-01-01 → 2024-10-31 | ~126,090 |
| Validación (early stopping) | 2024-11-01 → 2024-12-31 | ~13,078 |
| Test | 2025-01-01 → 2025-12-09 | ~57,344 |

Los modelos se entrenan con el target en escala `log1p` y las predicciones se revierten con `expm1`. Esto es necesario porque LÍNEA A tiene una demanda media de ~30,000 pax/hora y LÍNEA H de ~130 — sin normalización, el modelo optimiza casi exclusivamente para las líneas grandes.

**Modelos entrenados:**

1. **Baseline** — tabla de lookup: promedio histórico por `(línea, hora, día_semana)` sobre el train completo. No usa features temporales ni lags; es el punto de referencia.
2. **RidgeBaseline** — regresión lineal regularizada (`Ridge(alpha=1.0)`) con `ColumnTransformer`: OHE para features categóricas (`linea_encoded`, `tipo_linea_encoded`) y `StandardScaler` + `SimpleImputer` para las numéricas. Incluido como baseline lineal para confirmar que el problema requiere modelos no lineales.
3. **Random Forest** — 200 árboles, profundidad máxima 15, entrenado en log-escala.
4. **LightGBM global** — 2000 estimadores, learning rate 0.02, early stopping en validación. Features categóricas explícitas para `linea_encoded`, `tipo_linea_encoded`, `dia_semana`, `mes`, `hora_del_dia`.
5. **XGBoost** — hiperparámetros optimizados con Optuna (50 trials, expanding window CV). Early stopping con `eval_metric=mae`. Ver detalles en la [sección de tuning](#hiperparámetros-xgboost--optuna-tuning).
6. **LightGBM por línea** — 12 modelos independientes (uno por línea). Guardados como dict serializado en `models_per_line.pkl`.

El script también expone un modo de walk-forward cross-validation independiente (flag `--cv`) que no escribe a `predictions.csv` ni `metrics.json`.

### Paso 4 — Dashboard (`04_dashboard.py`)

Genera un HTML autocontenido con 14 secciones usando Plotly. No requiere servidor ni conexión a internet. Todas las figuras están embebidas como JSON en el HTML.

Secciones del dashboard:
1. KPIs — modelo ganador (XGBoost) vs baseline
2. Comparativa de todos los modelos
3. Real vs predicho, series temporales (LightGBM — seleccionado por interpretabilidad)
4. Residuales por hora del día
5. Feature importance — LightGBM
6. Distribución del error por tipo de línea
7. MAE por línea individual
8. MAE por hora del día
9. **Festivos: MAE agrupado** (Baseline vs ML)
10. **Festivos: desglose por día individual**
11. **Tendencia interanual** (Jan–Sep comparable, 3 años)
12. **Comparación global vs equal-weight por línea**
13. Mejora porcentual de LGB_PerLinea vs baseline por línea
14. **★ Walk-forward Cross-Validation** — gráfico de línea por fold + tabla de resultados

### Paso 5 — AWS Upload (`05_aws_upload.py`)

Sube los 7 artefactos de output a `s3://metro-medellin-datalake/` usando `boto3`. Solo se ejecuta cuando `METRO_MODE=aws`; en modo local imprime un aviso y termina sin error. Artefactos subidos: `trusted.parquet`, `featured.parquet`, `predictions.csv`, `metrics.json`, `feature_importance.csv`, `metro_dashboard.html` (con `ContentType=text/html`) y `model_lgbm.pkl`.

---

## Resultados

### Métricas globales — Test set 2025

| Modelo | MAE | RMSE | R² | MAE equal-weight |
|--------|-----|------|----|-----------------|
| **Baseline** | **774.8** | 3,309 | 0.898 | **702.8** |
| **XGBoost** ⚡ | **839.3** | **3,416** | **0.892** | — |
| LGB_PerLinea | 887.4 | 3,587 | 0.880 | 802.5 |
| LightGBM | 890.2 | 3,617 | 0.878 | — |
| RandomForest | 905.4 | 3,930 | 0.856 | — |
| RidgeBaseline | 1,872.3 | 8,442 | 0.337 | — |

⚡ XGBoost tuneado con Optuna — ver [sección de hiperparámetros](#hiperparámetros-xgboost--optuna-tuning).

El baseline gana porque LÍNEA A (demanda promedio ~30k) domina el MAE global y es muy predecible con el promedio histórico. El MAE "equal-weight" pondera cada línea igual, pero el baseline sigue siendo el mejor.

### El argumento central: festivos

| Modelo | MAE días normales | MAE días festivos | Ratio |
|--------|------------------|------------------|-------|
| Baseline | 703 | **2,201** | 3.1× |
| LightGBM | 876 | **1,175** | 1.3× |

En los **10 festivos "quietos"** (demanda real 29–52% de lo normal) el baseline promedia MAE=2,751 y LightGBM promedia MAE=285. El ML es ~10 veces más preciso.

Los **4 festivos "activos"** (Ascensión 120%, Batalla de Boyacá 118%, Todos los Santos 112%, Inmaculada Concepción 132%) fallan en ambos modelos: el modelo aprendió que festivo implica menos demanda, y estos cuatro tienen demanda igual o mayor que un día normal por motivos cívicos o culturales.

### Walk-forward Cross-Validation (4 folds, ventana expandible)

| Fold | Período test | Baseline MAE | LightGBM MAE | Ganador |
|------|-------------|-------------|-------------|---------|
| 1 | Nov–Dic 2023 | 446.2 | 500.5 | Baseline |
| 2 | Mar–Jun 2024 | 640.6 | 454.2 | LightGBM |
| 3 | Jul–Sep 2024 | 420.2 | 362.1 | LightGBM |
| 4 | Oct–Dic 2024 | 557.9 | 371.1 | LightGBM |

LightGBM gana en 3 de 4 folds. El CV score (diferencia promedio como % del MAE medio) es 13.7%, por debajo del umbral de 15% establecido como criterio de estabilidad. El fold 1 (Baseline gana) corresponde al período de menor variabilidad estacional del sistema; la ventana de entrenamiento en ese fold es la más pequeña y no incluye ningún festivo "activo" representativo.

### Hiperparámetros XGBoost — Optuna tuning

El script `experiments/tune_hyperparams.py` corre 50 trials de Optuna sobre LightGBM y XGBoost usando expanding window CV (Fold 3 + Fold 4 del walkforward) como función objetivo. No modifica el pipeline ni escribe a `data/output/`.

**Resultado del experimento (rama `experiment/hyperparameter-tuning`):**

| Parámetro | Default | Tuneado | Efecto |
|-----------|---------|---------|--------|
| `max_depth` | 8 | **7** | Árbol ligeramente menos profundo — menor overfitting |
| `learning_rate` | 0.02 | **0.0995** | 5× más agresivo — converge en menos árboles |
| `n_estimators` | 2000 | **397** | 5× menos árboles — entrenamiento ~5× más rápido |
| `subsample` | 0.8 | **0.961** | Mayor uso de datos por árbol |
| `colsample_bytree` | 0.8 | **0.713** | Menos features por árbol — más regularización |

**Impacto en el test set 2025:**

| Métrica | Antes (defaults) | Después (tuneado) | Δ |
|---------|-----------------|-------------------|---|
| MAE global | 865.9 | **839.3** | −3.07% |
| RMSE | 3,527 | **3,416** | −3.15% |
| R² | 0.884 | **0.892** | +0.008 |
| MAE festivos | 1,174.9 | **1,141.1** | −2.88% |

LightGBM empeoró con tuning (−6.2%): Optuna encontró `learning_rate=0.216` con 173 árboles, configuración que funciona bien en los folds 2024 pero no generaliza al test 2025 como el modelo con `lr=0.02` y 2000 árboles con early stopping.

```bash
# Reproducir el experimento (requiere featured.parquet generado)
python experiments/tune_hyperparams.py
```

---

## Hallazgos del EDA y del proceso

### H1 — Bug de fechas mixto en el CSV fuente

El archivo original usa formato D/M/YYYY para registros de 2023 y 2024, pero M/D/YYYY para 2025. No está documentado en ningún lado y no es configurable con un único parámetro de `pd.to_datetime`. La solución fue una doble pasada: primero parseo con `dayfirst=True`, luego una segunda pasada con `dayfirst=False` únicamente sobre los registros que quedaron NaN. Resultado: 0 fechas sin parsear en 236,220 registros.

### H2 — Tendencia interanual: el sistema está en declive

Comparando solo los meses enero–septiembre (los únicos completos en los tres años), la afluencia cayó:

| Período | Total pax ene–sep | Variación |
|---------|------------------|-----------|
| 2023 | 254,433,388 | — |
| 2024 | 247,033,063 | -2.91% |
| 2025 | 226,257,916 | -8.41% |

Comparar totales anuales brutos daría cifras erróneas porque octubre–diciembre 2025 solo tienen datos parciales (el dataset llega hasta 2025-12-09). La feature `año_trend = año - 2023` permite a los modelos capturar esta tendencia.

### H3 — Outlier: LÍNEA A, 14 de diciembre de 2023

Ese día se registraron 165,612 pasajeros en una hora —aproximadamente el doble de la media de LÍNEA A. El patrón horario es coherente (no es un pico puntual de un slot), lo que sugiere un evento masivo en el área de influencia (centro de Medellín). Se conservó en el dataset; eliminar un outlier real introduciría más sesgo que mantenerlo.

### H4 — LÍNEA L: 50.3% de horas con NaN estructural

LÍNEA L opera en horario muy reducido. Las horas fuera de operación no son errores de captura sino ausencias reales. El tratamiento correcto es diferenciarlas de los NaN incidentales y llenarlas con 0 solo después de calcular los lags, para no propagar el cero artificial como si fuera demanda real.

### H5 — Contaminación de lags por el orden de operaciones

En la primera versión del pipeline, el relleno NaN→0 ocurría antes del cálculo de lags. El efecto: el `lag_1d` del lunes de LÍNEA L tomaba el 0 del domingo (hora no operativa), no el valor del lunes anterior. Esto subestimaba sistemáticamente la demanda predicha para esas líneas al inicio de semana. Al invertir el orden (lags primero, relleno después) se perdieron 393 registros en el test set (primeras filas sin suficiente historial), un costo menor al sesgo que corregía.

### H6 — Los modelos por línea no superan al baseline global

Se entrenaron 12 LightGBM independientes esperando que cada uno capturara mejor los patrones específicos de su línea. Solo 3 de 12 mejoran: LÍNEA H (+62%), LÍNEA L (+21%) y LÍNEA O (+12%). Las 9 líneas restantes, incluyendo LÍNEA A que domina el MAE global, se predicen mejor con el promedio histórico. La conclusión es que para líneas regulares y con alta periodicidad semanal, la especialización no añade valor.

### H7 — `dia_del_año` no introduce leakage

Se sospechaba que incluir el día del año podía crear leakage estacional (el modelo "aprendería" que el día 1 del año tiene demanda X sin generalizar). El ablation test (entrenar sin esa feature) mostró que el MAE empeora +7.2 puntos al quitarla. No hay leakage: la feature captura estacionalidad real sin memorizar fechas específicas.

### H8 — RidgeBaseline confirma no-linealidad inter-línea

Ridge Regression con OHE por línea alcanzó MAE=1,872, peor que el propio Baseline histórico (774). Incluso con `ColumnTransformer` que expone cada línea como variable dummy independiente, Ridge aprende un único coeficiente global para `lag_7d`. La interacción `lag_7d × línea` —que es exactamente lo que captura LGB_PerLinea con sus 12 modelos independientes— es inherentemente no lineal. Este resultado confirma que la estructura del problema requiere modelos de árbol o modelos separados por línea; un enfoque lineal global no es suficiente.

---

## Asunciones y decisiones de diseño

**Split temporal estricto.** Train termina en octubre 2024, validación en noviembre–diciembre 2024, test en todo 2025. No se usó validación cruzada porque mezclaría fechas futuras con pasadas, inflando artificialmente las métricas.

**Log1p en el target.** La diferencia de escala entre LÍNEA A (~30k pax/hora) y LÍNEA H (~130 pax/hora) es de 230×. Sin normalización, los modelos minimizarían el error solo en las líneas grandes. La transformación logarítmica balancea el aprendizaje entre líneas sin mezclar sus series.

**Festivo vs. puente.** Se usó la librería `holidays.Colombia` que aplica automáticamente la Ley de Puentes (traslado al lunes siguiente para ciertos festivos). La feature `es_festivo` refleja el día oficial, y `es_puente` se deriva de los lunes resultantes de ese traslado. No se añadió una categoría de festivo (cívico, religioso, etc.) porque los datos de entrenamiento de 2023–2024 son insuficientes para aprender esa distinción de forma fiable.

**Diseño de RidgeBaseline.** Se usó `ColumnTransformer` con OHE para `linea_encoded` y `tipo_linea_encoded` en lugar de pasarlos como enteros, porque tratarlos como ordinales introduciría una relación lineal espuria (línea 0 "más cerca" de línea 1 que de línea 11). Con OHE, Ridge puede aprender interceptos por línea. Aun así, los coeficientes de lag_7d y rolling_7d son globales — la no-linealidad entre líneas no es capturable sin interacciones explícitas o modelos separados.

**Modelo ganador para el KPI del dashboard.** El dashboard muestra XGBoost como ganador porque tiene el menor MAE global entre los modelos ML (839.3 post-tuning). La sección de visualización temporal usa LightGBM porque su feature importance es interpretable (el modelo expone `gain` por feature de forma nativa), lo cual es más valioso para la presentación académica que XGBoost en esa sección.

**metrics.json generado desde predictions.csv.** Tras guardar `predictions.csv` a disco, las métricas se recalculan leyendo ese archivo y `featured.parquet`. Esto garantiza que el JSON siempre es consistente con las predicciones escritas y puede regenerarse de forma independiente.

**Compatibilidad AWS EMR.** Los scripts usan `os.getenv("METRO_MODE", "local")` y rutas relativas al directorio de trabajo para permitir ejecución tanto en local como en un cluster EMR sin modificar código. Los modelos se serializan con `pickle` en formato compatible con Python 3.8+. `05_aws_upload.py` es un no-op en modo local para que `run_pipeline.py` no requiera credenciales AWS en desarrollo.

---

## Complicaciones encontradas

**Codificación en Windows.** La consola de Windows usa cp1252 por defecto. Los prints con tildes y caracteres especiales fallaban silenciosamente o producían UnicodeEncodeError. Se resolvió con `python -X utf8` al invocar cada script y añadiendo `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` al inicio de cada módulo.

**pip con bytes nulos.** En el entorno de desarrollo, el ejecutable `pip.exe` tenía bytes nulos que causaban SyntaxError al invocarlo directamente. La solución fue usar `python -m pip install` en lugar de llamar `pip` como comando independiente.

**`add_vline` de Plotly en versiones recientes.** La llamada `fig.add_vline()` lanzaba `TypeError` en la versión instalada. Se reemplazó con la API de bajo nivel: `fig.add_shape(type="line")` más `fig.add_annotation()` para la etiqueta.

**Filas duplicadas en el CSV fuente.** Durante el EDA apareció lo que parecía un registro duplicado de LÍNEA B para el 1 de enero de 2024. Tras inspección resultó ser el registro del 1 de enero de 2023 con fecha mal capturada en el CSV original. Se corrigió directamente en el archivo fuente; no requirió cambios en el código.

**NaN en Ridge por features de lag.** `hacer_split` descarta NaN solo en las columnas `["pasajeros", "lag_1d", "lag_7d", "rolling_7d"]`, dejando NaN en `lag_14d` y `rolling_28d` para las primeras filas de cada serie. Ridge lanzaba `ValueError: Input X contains NaN`. Se resolvió incluyendo `SimpleImputer(strategy="median")` en el `Pipeline` de preprocesamiento.

**Métricas de festivos con pocas muestras.** Algunos festivos tienen muy pocos registros en el test (especialmente los de noviembre y diciembre 2025, donde el dataset llega solo hasta el 9 de diciembre). Las métricas por día individual deben interpretarse con cautela para esos casos; por eso se añade `n_registros` en el JSON de métricas.

---

## Dependencias

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
plotly>=5.15.0
holidays>=0.40
pyarrow>=12.0.0
openpyxl>=3.1.0
```

Python 3.9 o superior recomendado.
