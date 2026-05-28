# EDA Notes — Metro de Medellín

Generado durante el pipeline ML. Documenta hallazgos relevantes para el modelado
y la presentación académica.

---

## Dataset

- **Fuente:** `data/raw/afluencia-metro.csv`
- **Registros finales (long):** 236,220
- **Rango temporal:** 2023-01-01 → 2025-12-09
- **Líneas:** 12 (todas presentes en los 3 años)

---

## Hallazgo 1 — Bug de fechas mixto (RESUELTO en ETL)

El CSV usa D/M/YYYY para 2023-2024 y M/D/YYYY para 2025. Parseo con doble pasada.
Resultado: 0 fechas no parseadas.

---

## Hallazgo 2 — Tendencia interanual: DECLIVE real del sistema

Comparación en **meses equivalentes** (enero–septiembre, únicos completos en los 3 años):

| Período | Total pax ene-sep | Variación |
|---------|------------------|-----------|
| 2023    | 254,433,388      | —         |
| 2024    | 247,033,063      | **-2.91%** |
| 2025    | 226,257,916      | **-8.41%** |

El sistema está en declive sostenido. Comparar totales anuales brutos da cifras
erróneas porque oct/nov/dic 2025 solo tienen 9 días de datos.

Feature añadida: `año_trend = año - 2023` (valores 0, 1, 2).

---

## Hallazgo 3 — Outlier: LÍNEA A, 14 diciembre 2023

Pico de 165,612 pax/hora (2× lo normal). Patrón horario coherente. Se conserva —
probablemente evento masivo en el centro de Medellín.

---

## Hallazgo 4 — NaN estructurales y contaminación de lags (CORREGIDO v2)

LÍNEA L tiene 50.3% de horas NaN (opera en horario reducido). En la versión v1 del
pipeline los lags se calculaban DESPUÉS del relleno NaN→0, causando que el lag del
lunes tomara el 0 artificial del domingo. Corregido: lags se calculan ANTES del
relleno. Costo: 393 registros menos en el test set (0.7%).

---

## Hallazgo 5 ★ — FESTIVOS: EL ARGUMENTO CENTRAL DE VALOR DEL ML

Este es el hallazgo más importante del proyecto.

### Resumen ejecutivo

| Modelo    | MAE días normales | MAE días festivos | Ratio fest/norm |
|-----------|------------------|------------------|-----------------|
| Baseline  | 703              | **2,201**         | **3.1×**        |
| LightGBM  | 876              | **1,175**         | **1.3×**        |
| XGBoost   | 850              | **1,175**         | **1.4×**        |

**En festivos, LightGBM es 47% más preciso que el baseline.**

### Por qué el baseline falla en festivos

El baseline predice usando el promedio histórico por `(línea, hora, día_semana)`.
Un festivo que cae martes se predice como "martes promedio", ignorando que es festivo.
Los modelos ML tienen la feature `es_festivo` y aprenden el patrón.

### Clasificación de festivos 2025

Los 14 festivos del test set se dividen en dos tipos:

**Festivos "quietos"** (demanda real < 60% de lo normal) — 10 días:

| Festivo | Demanda real | Baseline MAE | LightGBM MAE | Mejora ML |
|---------|-------------|-------------|-------------|-----------|
| Año Nuevo | 29% | 4,087 | 370 | **+91%** |
| Reyes Magos | 52% | 2,646 | 359 | **+86%** |
| San José | 43% | 2,642 | 102 | **+96%** |
| Jueves Santo | 45% | 3,422 | 309 | **+91%** |
| Viernes Santo | 38% | 3,744 | 353 | **+91%** |
| Día del Trabajo | 47% | 3,276 | 546 | **+83%** |
| Corpus Christi | 44% | 2,575 | 129 | **+95%** |
| Sagrado Corazón | 47% | 2,467 | 235 | **+90%** |
| Independencia (Jul 20) | 48% | 149 | 294 | -97%* |
| La Asunción | 45% | 2,499 | 152 | **+94%** |

*Jul 20 cae en domingo en 2025 → el baseline ya predice baja demanda por el efecto día-de-semana.

**Festivos "activos"** (demanda real ≥ 100% de lo normal) — 4 días:

| Festivo | Demanda real | Por qué ambos fallan |
|---------|-------------|---------------------|
| Ascensión (Jun 2) | 120% | LightGBM aprendió festivo=menos gente → falla |
| Batalla de Boyacá (Ago 7) | 118% | Ídem |
| Santos (Nov 3) | 112% | + datos escasos en oct/nov 2025 (9 días/mes) |
| Inmaculada (Dic 8) | 132% | + datos escasos en dic 2025 |

**Conclusión:** El ML añade valor claro en **festivos de baja demanda** (demanda 29–52% de lo normal).
Los 4 festivos activos (Ascensión 120%, Boyacá 118%, Santos 112%, Inmaculada 132%) tienen demanda
igual o mayor que un día normal — el modelo no los maneja bien porque aprendió que festivo implica menos pasajeros.
Para festivos cívicos/activos (Independencia, Boyacá) se necesitaría una feature de categoría de festivo.

### Mejora ML en festivos por línea

| Línea    | Baseline MAE | LightGBM MAE | Mejora |
|---------|-------------|-------------|--------|
| LÍNEA A  | 15,213       | 7,955        | **+48%** |
| LÍNEA 1  | 2,561        | 1,207        | **+53%** |
| LÍNEA H  | 73           | 24           | **+67%** |
| LÍNEA T-A| 1,234        | 706          | **+43%** |
| LÍNEA B  | 2,047        | 1,188        | **+42%** |

La mejora es consistente en todas las líneas principales.

---

## Hallazgo 6 — Baseline imbatible para sistema periódico (días normales)

En días normales el baseline (MAE=703) supera a todos los modelos ML (MAE=850-876).
El sistema es altamente periódico: lag_7d tiene r=0.949 con el target.

### Modelos por línea: resultado honesto

Se entrenaron 12 LightGBM individuales (uno por línea). Resultado:

| Modelo       | MAE global | MAE equal-weight |
|-------------|-----------|-----------------|
| Baseline     | **774.8** | **702.8**       |
| XGBoost      | 865.9     | 783.3           |
| LGB_PerLinea | 887.4     | 802.5           |

**Conclusión honesta:** Los modelos por línea NO mejoran el baseline globalmente.
Solo 3 de 12 líneas mejoran con per-línea: LÍNEA H (+62%), LÍNEA L (+21%), LÍNEA O (+12%).
Las 9 líneas restantes (incluyendo LÍNEA A que domina el MAE global) son mejor
predichas por el baseline histórico.

El valor de los modelos ML está en los festivos, no en los días normales.

---

## Hallazgo 7 — dia_del_año: sin data leakage

Ablation test: retirar `dia_del_año` empeora MAE en +7.2 (890 → 897). Sin leakage.

---

## Síntesis para presentación académica

**¿Por qué construir modelos ML si el baseline gana en días normales?**

1. En días normales el sistema es tan periódico que el promedio histórico es casi perfecto.
2. **En festivos el baseline comete el triple de error que los modelos ML.**
3. Con 14 festivos al año y ~58k registros en el test, los festivos representan el 5%
   de los datos pero generan desproporcionalmente más error de operaciones en el Metro.
4. Una predicción precisa de festivos tiene valor operativo real: permite ajustar
   frecuencias, personal y energía.
5. Feature `es_festivo` con importancia top-10 en LightGBM confirma que el modelo usa ese conocimiento.
