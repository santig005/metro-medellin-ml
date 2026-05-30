"""
Dashboard ML — Metro de Medellín
Visualización de demanda, festivos y tendencias para decisiones operativas.
Desplegado en Streamlit Cloud desde el repo de GitHub.
"""

import json
from pathlib import Path

import holidays as holidays_lib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Metro Medellín — Dashboard ML",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paleta de colores
# ---------------------------------------------------------------------------
COLOR_REAL    = "#009EB0"   # teal  — demanda real
COLOR_PRED    = "#F5A623"   # gold  — predicción
COLOR_ERROR   = "#FF6B35"   # accent — errores
COLOR_DARK    = "#1C2B3A"   # fondo oscuro textos
COLOR_LIGHT   = "#F0F4F8"   # fondo claro

MODELOS_PRINCIPALES = ["Baseline", "XGBoost", "LightGBM", "RandomForest", "LGB_PerLinea"]

# ---------------------------------------------------------------------------
# Carga de datos con caché
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_predictions() -> pd.DataFrame:
    df = pd.read_csv(
        "data/output/predictions.csv",
        parse_dates=["fecha"],
        encoding="utf-8",
    )
    df["linea"] = df["linea"].str.strip()
    df["modelo"] = df["modelo"].str.strip()
    df["mes"] = df["fecha"].dt.to_period("M").astype(str)
    df["fecha_date"] = df["fecha"].dt.date
    return df


@st.cache_data
def cargar_metrics() -> dict:
    with open("data/output/metrics.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def calcular_festivos_df(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula MAE por festivo y modelo desde predictions.csv."""
    festivos_col = holidays_lib.Colombia(years=[2025])
    fechas_fest  = {str(k): v for k, v in festivos_col.items()}

    df_fest = df[df["fecha_date"].astype(str).isin(fechas_fest.keys())].copy()
    df_fest["festivo_nombre"] = df_fest["fecha_date"].astype(str).map(fechas_fest)

    resumen = (
        df_fest.groupby(["fecha_date", "festivo_nombre", "modelo"])
        .agg(
            mae=("error_absoluto", "mean"),
            total_real=("real", "sum"),
        )
        .reset_index()
    )
    resumen["fecha_date"] = pd.to_datetime(resumen["fecha_date"])
    resumen = resumen.sort_values("fecha_date")
    return resumen


@st.cache_data
def tabla_festivos_comparativa(df_fest: pd.DataFrame) -> pd.DataFrame:
    """Pivotea festivos para comparar Baseline vs XGBoost."""
    base = df_fest[df_fest["modelo"] == "Baseline"][["fecha_date", "festivo_nombre", "mae"]].rename(
        columns={"mae": "MAE Baseline"}
    )
    xgb = df_fest[df_fest["modelo"] == "XGBoost"][["fecha_date", "festivo_nombre", "mae"]].rename(
        columns={"mae": "MAE XGBoost"}
    )
    tabla = base.merge(xgb, on=["fecha_date", "festivo_nombre"], how="inner")
    tabla["Mejora %"] = (
        (tabla["MAE Baseline"] - tabla["MAE XGBoost"]) / tabla["MAE Baseline"] * 100
    ).round(1)
    tabla["MAE Baseline"] = tabla["MAE Baseline"].round(0).astype(int)
    tabla["MAE XGBoost"]  = tabla["MAE XGBoost"].round(0).astype(int)
    tabla["Fecha"] = tabla["fecha_date"].dt.strftime("%d/%m/%Y")
    return tabla[["Fecha", "festivo_nombre", "MAE Baseline", "MAE XGBoost", "Mejora %"]].rename(
        columns={"festivo_nombre": "Festivo"}
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        logo_path = Path("data/metro_medellin_logo.png")
        if logo_path.exists():
            st.image(str(logo_path), use_column_width=True)
        else:
            st.markdown("## 🚇 Metro Medellín")

        st.markdown("### Dashboard ML")
        st.markdown(
            "Herramienta de análisis predictivo para decisiones operativas. "
            "Modelo LightGBM entrenado sobre datos 2023–2024, evaluado en 2025."
        )
        st.divider()
        st.markdown(
            "<small>Datos: 2023–2025 · Modelos: 6 · Pipeline: EAFIT Git Hub</small>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sección 1 — Explorador de demanda
# ---------------------------------------------------------------------------
def seccion_explorador(df: pd.DataFrame):
    st.header("🔍 Explorador de demanda")
    st.caption(
        "Compara la demanda real con las predicciones del modelo para cualquier "
        "combinación de línea, fecha y hora."
    )

    lineas_disp = sorted(df["linea"].unique())
    modelos_disp = ["Baseline", "XGBoost", "LightGBM", "RandomForest", "LGB_PerLinea"]
    modelos_disp = [m for m in modelos_disp if m in df["modelo"].unique()]

    col_f1, col_f2, col_f3, col_f4 = st.columns([3, 2, 2, 2])
    with col_f1:
        lineas_sel = st.multiselect(
            "Líneas",
            lineas_disp,
            default=lineas_disp[:3],
            key="s1_lineas",
        )
    with col_f2:
        fecha_ini = st.date_input(
            "Desde",
            value=pd.Timestamp("2025-01-01").date(),
            min_value=df["fecha"].min().date(),
            max_value=df["fecha"].max().date(),
            key="s1_fecha_ini",
        )
    with col_f3:
        fecha_fin = st.date_input(
            "Hasta",
            value=pd.Timestamp("2025-03-31").date(),
            min_value=df["fecha"].min().date(),
            max_value=df["fecha"].max().date(),
            key="s1_fecha_fin",
        )
    with col_f4:
        modelo_sel = st.selectbox(
            "Modelo",
            modelos_disp,
            index=modelos_disp.index("XGBoost") if "XGBoost" in modelos_disp else 0,
            key="s1_modelo",
        )

    rango_hora = st.slider(
        "Rango de horas",
        min_value=4, max_value=23,
        value=(6, 22),
        key="s1_horas",
    )

    if not lineas_sel:
        st.warning("Selecciona al menos una línea.")
        return

    mask = (
        df["linea"].isin(lineas_sel)
        & (df["fecha"].dt.date >= fecha_ini)
        & (df["fecha"].dt.date <= fecha_fin)
        & (df["hora"] >= rango_hora[0])
        & (df["hora"] <= rango_hora[1])
        & (df["modelo"] == modelo_sel)
    )
    df_fil = df[mask].copy()

    if df_fil.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return

    # — Métricas resumen —
    mae_fil  = df_fil["error_absoluto"].mean()
    total_px = df_fil["real"].sum()
    hora_pico = (
        df_fil.groupby("hora")["real"].mean().idxmax()
        if not df_fil.empty else 0
    )
    linea_pico = (
        df_fil.groupby("linea")["real"].sum().idxmax()
        if not df_fil.empty else "-"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE del modelo", f"{mae_fil:,.0f} pax")
    m2.metric("Total pasajeros (período)", f"{total_px:,.0f}")
    m3.metric("Hora pico promedio", f"{hora_pico}:00 h")
    m4.metric("Línea más demandada", linea_pico)

    # — Gráfico real vs predicho —
    df_plot = (
        df_fil.groupby(["fecha", "linea"])[["real", "prediccion"]]
        .mean()
        .reset_index()
    )

    fig = go.Figure()
    for linea in lineas_sel:
        sub = df_plot[df_plot["linea"] == linea]
        fig.add_trace(go.Scatter(
            x=sub["fecha"], y=sub["real"],
            name=f"{linea} — Real",
            mode="lines",
            line=dict(color=COLOR_REAL, width=1.5),
            opacity=0.85,
            legendgroup=linea,
        ))
        fig.add_trace(go.Scatter(
            x=sub["fecha"], y=sub["prediccion"],
            name=f"{linea} — Predicción ({modelo_sel})",
            mode="lines",
            line=dict(color=COLOR_PRED, width=1.5, dash="dot"),
            opacity=0.85,
            legendgroup=linea,
        ))

    fig.update_layout(
        title=dict(
            text=f"Demanda real vs predicha — {modelo_sel}",
            y=0.97,
            x=0,
            xanchor="left",
            yanchor="top",
            font=dict(size=16),
        ),
        xaxis_title="Fecha",
        yaxis_title="Pasajeros promedio / hora",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        hovermode="x unified",
        height=450,
        template="plotly_white",
        margin=dict(t=60, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver datos detallados"):
        st.dataframe(
            df_fil[["fecha", "linea", "hora", "real", "prediccion", "error_absoluto"]]
            .sort_values(["fecha", "linea", "hora"])
            .reset_index(drop=True),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Sección 2 — Festivos críticos
# ---------------------------------------------------------------------------
def seccion_festivos(df: pd.DataFrame):
    st.header("🎉 Festivos críticos")
    st.caption(
        "En días festivos los patrones de demanda cambian radicalmente. "
        "El modelo XGBoost reduce el error hasta un 97% respecto al Baseline histórico."
    )

    # Filtros
    col_a, col_b = st.columns([2, 3])
    with col_a:
        año_sel = st.selectbox("Año", [2025], key="s2_year")
    with col_b:
        lineas_disp = sorted(df["linea"].unique())
        lineas_fest = st.multiselect(
            "Filtrar por línea (vacío = todas)",
            lineas_disp,
            default=[],
            key="s2_lineas",
        )

    df_base = df[df["modelo"].isin(["Baseline", "XGBoost"])]
    if lineas_fest:
        df_base = df_base[df_base["linea"].isin(lineas_fest)]

    df_fest   = calcular_festivos_df(df_base)
    tabla_cmp = tabla_festivos_comparativa(df_fest)

    if tabla_cmp.empty:
        st.info("No hay datos de festivos para los filtros seleccionados.")
        return

    # — Destacar Año Nuevo —
    año_nuevo = tabla_cmp[tabla_cmp["Fecha"] == "01/01/2025"]
    if not año_nuevo.empty:
        fila_an = año_nuevo.iloc[0]
        st.markdown("#### ⭐ Año Nuevo 2025 — caso emblemático")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "MAE Baseline (promedio histórico)",
            f"{fila_an['MAE Baseline']:,} pax",
        )
        c2.metric(
            "MAE XGBoost",
            f"{fila_an['MAE XGBoost']:,} pax",
            delta=f"−{fila_an['MAE Baseline'] - fila_an['MAE XGBoost']:,} pax",
            delta_color="inverse",
        )
        c3.metric(
            "Reducción de error",
            f"{fila_an['Mejora %']:.1f}%",
            delta="vs Baseline histórico",
            delta_color="inverse",
        )
        st.info(
            "📌 **¿Por qué importa?** El Baseline asume que Año Nuevo se comporta como un día "
            "normal del mismo día de semana. XGBoost aprendió que la demanda colapsa en festivos "
            "de alto impacto — información crítica para no despachar trenes vacíos."
        )
        st.divider()

    # — Tabla comparativa —
    st.markdown("#### Comparativa Baseline vs XGBoost por festivo")

    def colorear_mejora(val):
        if isinstance(val, (int, float)):
            if val >= 50:
                return "background-color: #d4edda; color: #155724"
            elif val >= 20:
                return "background-color: #fff3cd; color: #856404"
            else:
                return "background-color: #f8d7da; color: #721c24"
        return ""

    styled = tabla_cmp.style.map(colorear_mejora, subset=["Mejora %"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # — Gráfico de barras error por festivo —
    st.markdown("#### Error absoluto promedio por festivo")

    df_bar = df_fest[df_fest["modelo"].isin(["Baseline", "XGBoost"])].copy()
    df_bar["etiqueta"] = (
        df_bar["fecha_date"].dt.strftime("%d/%m") + " " + df_bar["festivo_nombre"].str[:20]
    )

    fig_bar = px.bar(
        df_bar.sort_values("fecha_date"),
        x="etiqueta",
        y="mae",
        color="modelo",
        barmode="group",
        color_discrete_map={"Baseline": COLOR_ERROR, "XGBoost": COLOR_PRED},
        labels={"mae": "MAE (pasajeros)", "etiqueta": "Festivo", "modelo": "Modelo"},
        title="MAE promedio por festivo — Baseline vs XGBoost",
        height=400,
    )
    fig_bar.update_layout(
        template="plotly_white",
        xaxis_tickangle=-35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=100),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------------------------
# Sección 3 — Tendencia y recomendaciones operativas
# ---------------------------------------------------------------------------
def seccion_tendencia(df: pd.DataFrame, metrics: dict):
    st.header("📉 Tendencia y recomendaciones operativas")
    st.caption(
        "La demanda del Metro presenta una tendencia decreciente sostenida desde 2023. "
        "Entender este patrón es clave para optimizar la oferta y reducir costos operativos."
    )

    # — Métricas de declive —
    st.markdown("#### Declive interanual de demanda")
    d1, d2, d3 = st.columns(3)
    d1.metric(
        "Variación 2023 → 2024",
        "−2.91%",
        delta="−2.91%",
        delta_color="inverse",
        help="Calculado sobre el total anual de pasajeros en datos históricos.",
    )
    d2.metric(
        "Variación 2024 → 2025",
        "−8.41%",
        delta="−8.41%",
        delta_color="inverse",
        help="Aceleración del declive observada en el set de prueba 2025.",
    )
    d3.metric(
        "Declive acumulado 2023→2025",
        "−11.07%",
        delta="−11.07%",
        delta_color="inverse",
        help="Caída total sobre el período completo del pipeline.",
    )

    # — Gráfico de área: tendencia mensual 2025 (real) —
    st.markdown("#### Demanda mensual 2025 — valores reales")

    df_real_all = df[df["modelo"] == "XGBoost"][["fecha", "real"]].copy()
    fecha_min = df_real_all["fecha"].min().date()
    fecha_max = df_real_all["fecha"].max().date()

    import datetime as _dt
    default_inicio = max(fecha_min, _dt.date(2025, 1, 1))
    default_fin    = min(fecha_max, _dt.date(2025, 9, 30))

    col_f1, col_f2 = st.columns(2)
    filtro_inicio = col_f1.date_input("Desde", value=default_inicio, min_value=fecha_min, max_value=fecha_max, key="tend_desde")
    filtro_fin    = col_f2.date_input("Hasta", value=default_fin,    min_value=fecha_min, max_value=fecha_max, key="tend_hasta")

    df_real = df_real_all[
        (df_real_all["fecha"].dt.date >= filtro_inicio) &
        (df_real_all["fecha"].dt.date <= filtro_fin)
    ].copy()

    df_mensual = (
        df_real.groupby(df_real["fecha"].dt.to_period("M"))["real"]
        .sum()
        .reset_index()
    )
    df_mensual["fecha_str"] = df_mensual["fecha"].astype(str)
    df_mensual["real"] = df_mensual["real"].round(0)

    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=df_mensual["fecha_str"],
        y=df_mensual["real"],
        fill="tozeroy",
        name="Pasajeros reales",
        line=dict(color=COLOR_REAL, width=2),
        fillcolor=f"rgba(0,158,176,0.25)",
        mode="lines+markers",
        marker=dict(size=6),
    ))

    # Línea de tendencia
    y_vals = df_mensual["real"].values
    x_idx  = np.arange(len(y_vals))
    if len(x_idx) > 1:
        coef = np.polyfit(x_idx, y_vals, 1)
        trend = np.polyval(coef, x_idx)
        fig_area.add_trace(go.Scatter(
            x=df_mensual["fecha_str"],
            y=trend,
            name="Tendencia lineal",
            line=dict(color=COLOR_ERROR, width=2, dash="dash"),
            mode="lines",
        ))

    fig_area.update_layout(
        title="Pasajeros totales por mes — 2025 (set de prueba)",
        xaxis_title="Mes",
        yaxis_title="Total pasajeros",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=60, b=80),
    )
    st.plotly_chart(fig_area, use_container_width=True)

    # — Tabla comparativa de modelos —
    st.markdown("#### Comparativa de modelos — Test set 2025")

    NOMBRES_DISPLAY = {
        "Baseline":     "Baseline histórico",
        "RidgeBaseline":"Ridge Regression",
        "RandomForest": "Random Forest",
        "LightGBM":     "LightGBM",
        "XGBoost":      "XGBoost",
        "LGB_PerLinea": "LGB por Línea",
    }
    filas = []
    baseline_mae = metrics.get("Baseline", {}).get("global", {}).get("mae", 1)
    for mod_key, display in NOMBRES_DISPLAY.items():
        g = metrics.get(mod_key, {}).get("global", {})
        if not g:
            continue
        mejora = round((baseline_mae - g["mae"]) / baseline_mae * 100, 1)
        filas.append({
            "Modelo":        display,
            "MAE":           g["mae"],
            "RMSE":          g["rmse"],
            "R²":            g["r2"],
            "Mejora vs Baseline": f"{mejora:+.1f}%",
        })

    df_modelos = pd.DataFrame(filas)

    def resaltar_mejor(s):
        if s.name == "MAE":
            idx_min = s.apply(lambda x: float(x)).idxmin()
            return ["background-color: #d4edda; font-weight: bold" if i == idx_min else "" for i in s.index]
        return [""] * len(s)

    st.dataframe(
        df_modelos.style.apply(resaltar_mejor, subset=["MAE"]),
        use_container_width=True,
        hide_index=True,
    )

    # — Bloque narrativo —
    st.markdown("#### 💡 Implicaciones operativas")
    st.info(
        """
        **El declive de demanda no es ruido — es una señal estructural.**

        La caída del **−8.41% en 2025** (frente al −2.91% de 2024) sugiere una aceleración
        en el abandono del Metro como modo de transporte principal. Las causas pueden incluir
        competencia del transporte informal, cambios en patrones de trabajo remoto o
        inseguridad percibida en estaciones.

        **Recomendaciones para el equipo operativo:**

        - 🕐 **Reducir frecuencia en horas valle** (10:00–15:00) en líneas de metrocable,
          donde el declive relativo es más pronunciado.
        - 📅 **Ajustar el plan de operación en festivos** según el modelo predictivo —
          el Baseline histórico sobredimensiona la oferta hasta en un 97% (Año Nuevo).
        - 📊 **Monitorear LÍNEA A y LÍNEA B** como indicadores líderes: el metro férreo
          concentra el mayor volumen y cualquier variación impacta la capacidad global.
        - 🔄 **Reentrenar el modelo cada trimestre** para capturar el declive continuo
          y mantener la precisión de las predicciones operativas.
        """
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    st.title("🚇 Metro de Medellín — Dashboard Predictivo ML")
    st.markdown(
        "**¿Cuándo y dónde reforzar la operación?** "
        "Análisis de demanda de pasajeros con modelos de Machine Learning "
        "para anticipar picos, reducir costos en horas valle y gestionar festivos críticos."
    )
    st.divider()

    with st.spinner("Cargando datos..."):
        df      = cargar_predictions()
        metrics = cargar_metrics()

    tab1, tab2, tab3 = st.tabs([
        "🔍 Explorador de demanda",
        "🎉 Festivos críticos",
        "📉 Tendencia y recomendaciones",
    ])

    with tab1:
        seccion_explorador(df)

    with tab2:
        seccion_festivos(df)

    with tab3:
        seccion_tendencia(df, metrics)


if __name__ == "__main__":
    main()
