"""
Dashboard interactivo HTML — Metro de Medellín
Genera data/output/metro_dashboard.html autocontenido (sin servidor).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

INPUT_FEATURED = "data/processed/featured.parquet"
INPUT_TRUSTED  = "data/processed/trusted.parquet"
INPUT_PRED     = "data/output/predictions.csv"
INPUT_METRICS  = "data/output/metrics.json"
INPUT_IMP      = "data/output/feature_importance.csv"
OUTPUT_HTML    = "data/output/metro_dashboard.html"

PALETA = {
    "azul":     "#1a5fcc",
    "rojo":     "#cc3a1a",
    "verde":    "#1acc6a",
    "naranja":  "#cc7a1a",
    "gris":     "#888888",
    "fondo":    "#0f1117",
    "panel":    "#1a1d27",
    "texto":    "#e0e0e0",
}

LAYOUT_BASE = dict(
    paper_bgcolor=PALETA["fondo"],
    plot_bgcolor=PALETA["panel"],
    font=dict(color=PALETA["texto"], family="Arial, sans-serif"),
    margin=dict(l=60, r=30, t=50, b=50),
)


# ---------------------------------------------------------------------------
# SECCIÓN 1 — KPIs header
# ---------------------------------------------------------------------------
def build_kpis(metrics: dict) -> go.Figure:
    """Tabla de KPIs del modelo ganador vs baseline."""
    # Determinar modelo ganador (mejor MAE entre los no-baseline)
    candidatos = {k: v for k, v in metrics.items() if k != "Baseline"}
    ganador_nombre = min(candidatos, key=lambda k: candidatos[k]["global"]["mae"])
    ganador = candidatos[ganador_nombre]
    base = metrics["Baseline"]["global"]

    fig = go.Figure()
    fig.add_trace(go.Table(
        header=dict(
            values=["Métrica", "Baseline", ganador_nombre, "Mejora"],
            fill_color=PALETA["azul"],
            font=dict(color="white", size=14, family="Arial"),
            align="center",
        ),
        cells=dict(
            values=[
                ["MAE", "RMSE", "R²"],
                [f"{base['mae']:,.1f}", f"{base['rmse']:,.1f}", f"{base['r2']:.4f}"],
                [
                    f"{ganador['global']['mae']:,.1f}",
                    f"{ganador['global']['rmse']:,.1f}",
                    f"{ganador['global']['r2']:.4f}",
                ],
                [
                    f"{ganador.get('mejora_mae_vs_baseline_pct', 0):+.1f}%",
                    f"{ganador.get('mejora_rmse_vs_baseline_pct', 0):+.1f}%",
                    "—",
                ],
            ],
            fill_color=[PALETA["panel"]] * 4,
            font=dict(color=PALETA["texto"], size=13),
            align="center",
            height=30,
        ),
    ))
    fig.update_layout(
        title=f"KPIs — Modelo ganador: {ganador_nombre} | Test set 2025",
        **LAYOUT_BASE,
        height=200,
    )
    return fig


# ---------------------------------------------------------------------------
# SECCIÓN 2 — Comparación de modelos
# ---------------------------------------------------------------------------
def build_comparacion_modelos(metrics: dict) -> go.Figure:
    nombres = list(metrics.keys())
    maes    = [m["global"]["mae"]  for m in metrics.values()]
    rmses   = [m["global"]["rmse"] for m in metrics.values()]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("MAE por modelo", "RMSE por modelo"))

    colores = [PALETA["gris"] if n == "Baseline" else PALETA["azul"] for n in nombres]
    fig.add_trace(go.Bar(x=nombres, y=maes, marker_color=colores, name="MAE",
                         text=[f"{v:,.0f}" for v in maes], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=nombres, y=rmses, marker_color=colores, name="RMSE",
                         text=[f"{v:,.0f}" for v in rmses], textposition="outside"), row=1, col=2)

    fig.update_layout(
        title="Comparación de modelos — Test set 2025",
        showlegend=False,
        **LAYOUT_BASE,
        height=380,
    )
    fig.update_yaxes(gridcolor="#333344")
    return fig


# ---------------------------------------------------------------------------
# SECCIÓN 3 — Real vs Predicho (líneas de mayor demanda)
# ---------------------------------------------------------------------------
def build_real_vs_pred(pred_df: pd.DataFrame, df_trusted: pd.DataFrame) -> go.Figure:
    # Las 3 líneas de mayor demanda total
    top3 = (
        df_trusted.groupby("linea")["pasajeros"].sum()
        .nlargest(3).index.tolist()
    )
    # Usar predicciones del modelo LightGBM
    modelo_viz = "LightGBM" if "LightGBM" in pred_df["modelo"].unique() else pred_df["modelo"].iloc[0]
    preds_lgb = pred_df[pred_df["modelo"] == modelo_viz].copy()

    # Agregar por día (suma de horas) para que la línea sea legible
    real_diario = (
        preds_lgb.groupby(["fecha", "linea"])[["real", "prediccion"]]
        .sum().reset_index()
    )

    colores_linea = [PALETA["azul"], PALETA["rojo"], PALETA["verde"]]
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f"LÍNEA {l.split()[-1]} — Real vs Predicho (diario, 2025)" for l in top3],
        vertical_spacing=0.1,
    )
    for i, linea in enumerate(top3):
        sub = real_diario[real_diario["linea"] == linea].sort_values("fecha")
        fig.add_trace(go.Scatter(
            x=sub["fecha"], y=sub["real"],
            name=f"{linea} Real", line=dict(color=PALETA["azul"], width=1.5),
            showlegend=(i == 0), legendgroup="real",
        ), row=i+1, col=1)
        fig.add_trace(go.Scatter(
            x=sub["fecha"], y=sub["prediccion"],
            name=f"{linea} Predicción", line=dict(color=PALETA["rojo"], width=1.5, dash="dot"),
            showlegend=(i == 0), legendgroup="pred",
        ), row=i+1, col=1)

    fig.update_layout(
        title=f"Real vs Predicho ({modelo_viz} — seleccionado por interpretabilidad y feature importance) — Top 3 líneas por demanda total",
        **LAYOUT_BASE,
        height=750,
    )
    fig.update_yaxes(gridcolor="#333344")
    return fig


# ---------------------------------------------------------------------------
# SECCIÓN 4 — Heatmap demanda por hora × día de semana (con dropdown de línea)
# ---------------------------------------------------------------------------
def build_heatmap(df: pd.DataFrame) -> go.Figure:
    lineas = sorted(df["linea"].unique())
    dias   = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    horas  = list(range(4, 24))

    # Construir heatmaps para cada línea
    data_por_linea = {}
    for linea in lineas:
        pivot = (
            df[df["linea"] == linea]
            .groupby(["hora_del_dia", "dia_semana"])["pasajeros"]
            .mean()
            .unstack(fill_value=0)
        )
        # Asegurar todas las horas y días
        pivot = pivot.reindex(index=horas, columns=range(7), fill_value=0)
        data_por_linea[linea] = pivot.values

    # Figura base con primera línea
    linea0 = lineas[0]
    fig = go.Figure(go.Heatmap(
        z=data_por_linea[linea0],
        x=dias,
        y=[str(h) for h in horas],
        colorscale="YlOrRd",
        colorbar=dict(title="Pax/hora"),
    ))

    # Botones dropdown
    botones = []
    for linea in lineas:
        botones.append(dict(
            method="restyle",
            label=linea,
            args=[{"z": [data_por_linea[linea]]}],
        ))

    fig.update_layout(
        title="Demanda promedio por hora × día de semana",
        updatemenus=[dict(
            type="dropdown",
            buttons=botones,
            x=1.0, y=1.15,
            xanchor="right",
        )],
        **LAYOUT_BASE,
        height=500,
        xaxis_title="Día de semana",
        yaxis_title="Hora del día",
    )
    return fig


# ---------------------------------------------------------------------------
# SECCIÓN 5 — Feature importance LightGBM
# ---------------------------------------------------------------------------
def build_feature_importance(imp_df: pd.DataFrame) -> go.Figure:
    top15 = imp_df.head(15).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=top15["importance"],
        y=top15["feature"],
        orientation="h",
        marker_color=PALETA["azul"],
        text=[f"{v:,.0f}" for v in top15["importance"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 15 features — LightGBM (gain)",
        **LAYOUT_BASE,
        height=500,
        xaxis_title="Importancia (gain)",
    )
    fig.update_xaxes(gridcolor="#333344")
    return fig


# ---------------------------------------------------------------------------
# SECCIÓN 6 — Distribución de errores por tipo de línea
# ---------------------------------------------------------------------------
def build_error_dist(pred_df: pd.DataFrame, df_trusted: pd.DataFrame) -> go.Figure:
    # Agregar tipo_linea a predicciones
    tipo_map = df_trusted[["linea", "tipo_linea"]].drop_duplicates().set_index("linea")["tipo_linea"]
    pred_lgb = pred_df[pred_df["modelo"] == "LightGBM"].copy()
    pred_lgb["tipo_linea"] = pred_lgb["linea"].map(tipo_map)

    tipos = sorted(pred_lgb["tipo_linea"].dropna().unique())
    colores = [PALETA["azul"], PALETA["rojo"], PALETA["verde"], PALETA["naranja"]]

    fig = go.Figure()
    for i, tipo in enumerate(tipos):
        sub = pred_lgb[pred_lgb["tipo_linea"] == tipo]["error_absoluto"]
        fig.add_trace(go.Histogram(
            x=sub,
            name=tipo,
            opacity=0.7,
            marker_color=colores[i % len(colores)],
            nbinsx=50,
        ))

    fig.update_layout(
        title="Distribución del error absoluto por tipo de línea (LightGBM, 2025)",
        barmode="overlay",
        **LAYOUT_BASE,
        height=420,
        xaxis_title="Error absoluto (pasajeros)",
        yaxis_title="Frecuencia",
    )
    fig.update_xaxes(gridcolor="#333344")
    fig.update_yaxes(gridcolor="#333344")
    return fig


# ---------------------------------------------------------------------------
# SECCIÓN 7 — EDA: Demanda total por línea
# ---------------------------------------------------------------------------
def build_eda_total_linea(df: pd.DataFrame) -> go.Figure:
    total = (
        df.groupby("linea")["pasajeros"].sum()
        .sort_values(ascending=True)
        .reset_index()
    )
    total["millones"] = total["pasajeros"] / 1_000_000

    fig = go.Figure(go.Bar(
        x=total["millones"],
        y=total["linea"],
        orientation="h",
        marker_color=PALETA["azul"],
        text=[f"{v:.1f}M" for v in total["millones"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Demanda total por línea — 2023-2025 (millones de pasajeros)",
        **LAYOUT_BASE,
        height=450,
        xaxis_title="Millones de pasajeros",
    )
    fig.update_xaxes(gridcolor="#333344")
    return fig


# ---------------------------------------------------------------------------
# SECCIÓN 8 — EDA: Serie temporal del sistema completo
# ---------------------------------------------------------------------------
def build_eda_serie_temporal(df: pd.DataFrame) -> go.Figure:
    total_diario = (
        df.groupby("fecha")["pasajeros"].sum().reset_index()
    )
    # Media móvil 7 días
    total_diario["ma7"] = total_diario["pasajeros"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=total_diario["fecha"], y=total_diario["pasajeros"],
        name="Total diario",
        line=dict(color=PALETA["azul"], width=1),
        opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=total_diario["fecha"], y=total_diario["ma7"],
        name="Media móvil 7 días",
        line=dict(color=PALETA["rojo"], width=2),
    ))
    # Línea vertical separando train y test (usando shape en lugar de vline por compatibilidad)
    fig.add_shape(
        type="line",
        x0="2025-01-01", x1="2025-01-01",
        y0=0, y1=1, yref="paper",
        line=dict(color=PALETA["naranja"], dash="dash", width=2),
    )
    fig.add_annotation(
        x="2025-01-01", y=1.0, yref="paper",
        text="Inicio test 2025", showarrow=False,
        font=dict(color=PALETA["naranja"]), xanchor="left",
    )
    fig.update_layout(
        title="Total diario de pasajeros del sistema — 2023-2025",
        **LAYOUT_BASE,
        height=420,
        xaxis_title="Fecha",
        yaxis_title="Pasajeros totales",
        legend=dict(x=0.01, y=0.99),
    )
    fig.update_yaxes(gridcolor="#333344")
    fig.update_xaxes(gridcolor="#333344")
    return fig


# ---------------------------------------------------------------------------
# SECCIÓN 9 — Festivos: el argumento central de valor del ML
# ---------------------------------------------------------------------------
def build_festivos(metrics: dict, pred_df: pd.DataFrame) -> go.Figure:
    """
    Muestra MAE en días normales vs festivos para cada modelo.
    Argumento: en festivos 'quietos' el baseline falla 3× más que en días normales,
    mientras los modelos ML mantienen el error controlado.
    """
    import holidays as holidays_lib
    festivos_col = holidays_lib.Colombia(years=[2025])
    fechas_festivas = set(festivos_col.keys())

    pred_df = pred_df.copy()
    pred_df["es_festivo"] = pred_df["fecha"].dt.date.apply(lambda d: d in fechas_festivas).astype(int)

    modelos_ord = ["Baseline", "XGBoost", "LightGBM", "LGB_PerLinea"]
    modelos_ord = [m for m in modelos_ord if m in pred_df["modelo"].unique()]

    mae_normal = []
    mae_festivo = []
    for mod in modelos_ord:
        sub = pred_df[pred_df["modelo"] == mod]
        from sklearn.metrics import mean_absolute_error
        mae_n = mean_absolute_error(sub[sub["es_festivo"]==0]["real"],
                                    sub[sub["es_festivo"]==0]["prediccion"])
        mae_f = mean_absolute_error(sub[sub["es_festivo"]==1]["real"],
                                    sub[sub["es_festivo"]==1]["prediccion"])
        mae_normal.append(mae_n)
        mae_festivo.append(mae_f)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Días normales",
        x=modelos_ord, y=mae_normal,
        marker_color=PALETA["azul"],
        text=[f"{v:,.0f}" for v in mae_normal],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Días festivos",
        x=modelos_ord, y=mae_festivo,
        marker_color=PALETA["rojo"],
        text=[f"{v:,.0f}" for v in mae_festivo],
        textposition="outside",
    ))
    fig.update_layout(
        title="MAE en días normales vs festivos<br>"
              "<sup>En festivos de baja demanda (29–52% de lo normal) ML gana ~10× | En festivos activos (>100%) ambos modelos fallan</sup>",
        barmode="group",
        **LAYOUT_BASE,
        height=420,
        yaxis_title="MAE (pasajeros/hora)",
        legend=dict(x=0.7, y=0.95),
    )
    fig.update_yaxes(gridcolor="#333344")
    return fig


def build_festivos_por_dia(pred_df: pd.DataFrame) -> go.Figure:
    """MAE por día festivo individual: Baseline vs LightGBM."""
    import holidays as holidays_lib
    from sklearn.metrics import mean_absolute_error
    festivos_col = holidays_lib.Colombia(years=[2025])
    fechas_festivas = set(festivos_col.keys())

    pred_df = pred_df.copy()
    pred_df["es_festivo"] = pred_df["fecha"].dt.date.apply(lambda d: d in fechas_festivas).astype(int)
    fest_days = sorted(pred_df[pred_df["es_festivo"]==1]["fecha"].dt.date.unique())

    base = pred_df[pred_df["modelo"]=="Baseline"]
    lgb  = pred_df[pred_df["modelo"]=="LightGBM"]

    nombres, maes_b, maes_l, tipos = [], [], [], []
    media_normal = pred_df[pred_df["es_festivo"]==0]["real"].mean()

    for d in fest_days:
        b = base[base["fecha"].dt.date == d]
        l = lgb[ lgb["fecha"].dt.date  == d]
        if len(b) < 5:
            continue
        pct_real = b["real"].mean() / media_normal * 100
        tipo = "Activo (>60%)" if pct_real >= 60 else "Quieto (<60%)"
        nombre = festivos_col.get(d, str(d))[:25]
        nombres.append(f"{d.strftime('%d/%m')} {nombre}")
        maes_b.append(mean_absolute_error(b["real"], b["prediccion"]))
        maes_l.append(mean_absolute_error(l["real"], l["prediccion"]))
        tipos.append(tipo)

    df_plot = pd.DataFrame({"dia": nombres, "Baseline": maes_b, "LightGBM": maes_l, "tipo": tipos})
    df_plot = df_plot.sort_values("Baseline", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline",
        y=df_plot["dia"], x=df_plot["Baseline"],
        orientation="h", marker_color=PALETA["gris"],
    ))
    fig.add_trace(go.Bar(
        name="LightGBM",
        y=df_plot["dia"], x=df_plot["LightGBM"],
        orientation="h", marker_color=PALETA["azul"],
    ))
    fig.update_layout(
        title="MAE por día festivo — Baseline vs LightGBM<br>"
              "<sup>Festivos de baja demanda (29–52% normal): ML gana 83-96% | Festivos de alta demanda (Boyacá 118%, Ascensión 120%, Santos 112%): ambos fallan</sup>",
        barmode="group",
        **LAYOUT_BASE,
        height=520,
        xaxis_title="MAE",
        legend=dict(x=0.7, y=0.05),
    )
    fig.update_xaxes(gridcolor="#333344")
    return fig


# ---------------------------------------------------------------------------
# SECCIÓN 10 — Tendencia real de demanda anual (con el declive correcto)
# ---------------------------------------------------------------------------
def build_tendencia_anual(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["año"]  = pd.to_datetime(df["fecha"]).dt.year
    df["mes"]  = pd.to_datetime(df["fecha"]).dt.month

    # Solo meses con datos en los 3 años (1–9 = enero a septiembre completos)
    meses_ok = list(range(1, 10))
    sub = df[df["mes"].isin(meses_ok)]
    anual = sub.groupby("año")["pasajeros"].sum().reset_index()
    anual["millones"] = anual["pasajeros"] / 1_000_000

    # Variación YoY
    cambios = []
    for i in range(len(anual)):
        if i == 0:
            cambios.append("")
        else:
            pct = (anual["millones"].iloc[i] / anual["millones"].iloc[i-1] - 1) * 100
            cambios.append(f"{pct:+.1f}%")
    anual["cambio"] = cambios

    colores = [PALETA["azul"] if v == 0 else (PALETA["rojo"] if v < 0 else PALETA["verde"])
               for v in [0, -1, -1]]   # todos negativos excepto 2023 (base)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=anual["año"].astype(str),
        y=anual["millones"],
        marker_color=[PALETA["azul"], PALETA["rojo"], PALETA["rojo"]],
        text=[f"{v:.1f}M {c}" for v, c in zip(anual["millones"], anual["cambio"])],
        textposition="outside",
        width=0.5,
    ))

    fig.update_layout(
        title="Demanda del sistema: ene–sep 2023 vs 2024 vs 2025 (meses comparables)<br>"
              "<sup>Variación: 2023→2024 −2.91% | 2024→2025 −8.41% — el sistema está en declive</sup>",
        **LAYOUT_BASE,
        height=380,
        xaxis_title="Año",
        yaxis_title="Millones de pasajeros (ene–sep)",
    )
    fig.update_yaxes(gridcolor="#333344")
    return fig


# ---------------------------------------------------------------------------
# SECCIÓN 10 — Comparación modelos globales vs per-línea
# ---------------------------------------------------------------------------
def build_comparacion_perline(metrics: dict) -> go.Figure:
    """Tabla + gráfico con métricas globales y equal-weight per-line MAE."""
    modelos_ord = ["Baseline", "XGBoost", "LightGBM", "LGB_PerLinea"]
    modelos_ord = [m for m in modelos_ord if m in metrics]

    mae_global    = [metrics[m]["global"]["mae"]  for m in modelos_ord]
    rmse_global   = [metrics[m]["global"]["rmse"] for m in modelos_ord]
    r2_global     = [metrics[m]["global"]["r2"]   for m in modelos_ord]

    # Equal-weight MAE (cada línea cuenta igual)
    mae_equalw = []
    for m in modelos_ord:
        maes = [v["mae"] for v in metrics[m]["por_linea"].values()]
        mae_equalw.append(sum(maes) / len(maes))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("MAE global (dominado por Línea A)", "MAE equal-weight (cada línea = 1 voto)"),
    )
    colores = [PALETA["gris"] if n == "Baseline" else PALETA["azul"] for n in modelos_ord]

    fig.add_trace(go.Bar(
        x=modelos_ord, y=mae_global,
        marker_color=colores,
        text=[f"{v:,.0f}" for v in mae_global],
        textposition="outside", name="MAE global",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=modelos_ord, y=mae_equalw,
        marker_color=colores,
        text=[f"{v:,.0f}" for v in mae_equalw],
        textposition="outside", name="MAE eq-weight",
    ), row=1, col=2)

    fig.update_layout(
        title="Comparación justa de modelos: MAE global vs equal-weight por línea",
        showlegend=False,
        **LAYOUT_BASE,
        height=420,
    )
    fig.update_yaxes(gridcolor="#333344")

    # Segundo bloque: mejoras del modelo per-line sobre baseline por línea
    fig2 = go.Figure()
    base_pl = metrics["Baseline"]["por_linea"]
    if "LGB_PerLinea" in metrics:
        ml_pl = metrics["LGB_PerLinea"]["por_linea"]
        lineas_sorted = sorted(base_pl.keys())
        mejoras = [(base_pl[l]["mae"] - ml_pl.get(l, {}).get("mae", base_pl[l]["mae"])) /
                   base_pl[l]["mae"] * 100 for l in lineas_sorted]

        fig2.add_trace(go.Bar(
            x=lineas_sorted, y=mejoras,
            marker_color=[PALETA["verde"] if v > 0 else PALETA["rojo"] for v in mejoras],
            text=[f"{v:+.1f}%" for v in mejoras],
            textposition="outside",
        ))
        fig2.update_layout(
            title="% mejora LGB_PerLinea vs Baseline en MAE por línea<br>"
                  "<sup>Verde = ML gana, Rojo = baseline sigue siendo mejor</sup>",
            **LAYOUT_BASE,
            height=380,
            yaxis_title="% mejora en MAE",
        )
        fig2.update_yaxes(gridcolor="#333344")

    return fig, fig2


# ---------------------------------------------------------------------------
# SECCIÓN 14 — Walk-forward Cross-Validation
# Resultados hardcodeados desde la ejecución de walkforward_cv() en 03_models.py.
# ---------------------------------------------------------------------------
def build_walkforward_cv() -> go.Figure:
    folds = ["Fold 1\nene–jun 2023", "Fold 2\nene–dic 2023",
             "Fold 3\nene–jun 2024", "Fold 4\nene–sep 2024"]
    periodos_test = ["jul–sep 2023", "ene–mar 2024", "jul–sep 2024", "oct–dic 2024"]
    mae_base = [446.2, 640.6, 420.2, 557.9]
    mae_lgb  = [500.5, 454.2, 362.1, 371.1]
    ganador  = ["Baseline", "LightGBM", "LightGBM", "LightGBM"]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "MAE por fold — Baseline vs LightGBM",
            "Tabla de resultados por fold",
        ),
        vertical_spacing=0.18,
        row_heights=[0.55, 0.45],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )

    # Gráfico de líneas
    x_idx = list(range(1, 5))
    fig.add_trace(go.Scatter(
        x=x_idx, y=mae_base,
        mode="lines+markers+text",
        name="Baseline",
        line=dict(color=PALETA["gris"], width=2, dash="dash"),
        marker=dict(size=10),
        text=[f"{v:,.0f}" for v in mae_base],
        textposition="top right",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_idx, y=mae_lgb,
        mode="lines+markers+text",
        name="LightGBM",
        line=dict(color=PALETA["azul"], width=2),
        marker=dict(size=10),
        text=[f"{v:,.0f}" for v in mae_lgb],
        textposition="bottom right",
    ), row=1, col=1)

    # Área sombreada donde LightGBM gana (Fold 2-4)
    fig.add_shape(type="rect",
        x0=1.5, x1=4.5, y0=0, y1=1,
        xref="x", yref="paper",
        fillcolor=PALETA["azul"], opacity=0.07, line_width=0,
        row=1, col=1,
    )
    fig.update_xaxes(
        tickvals=x_idx,
        ticktext=[f"F{i}" for i in x_idx],
        row=1, col=1,
    )
    fig.update_yaxes(gridcolor="#333344", title_text="MAE (pasajeros/hora)", row=1, col=1)

    # Tabla de resultados
    colores_ganador = [PALETA["gris"] if g == "Baseline" else PALETA["azul"] for g in ganador]
    fig.add_trace(go.Table(
        header=dict(
            values=["Fold", "Período test", "Train hasta", "Baseline MAE", "LightGBM MAE", "Ganador"],
            fill_color=PALETA["azul"],
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[
                [f"Fold {i}" for i in range(1, 5)],
                periodos_test,
                ["jun 2023", "dic 2023", "jun 2024", "sep 2024"],
                [f"{v:,.1f}" for v in mae_base],
                [f"{v:,.1f}" for v in mae_lgb],
                ganador,
            ],
            fill_color=[
                [PALETA["panel"]] * 4,
                [PALETA["panel"]] * 4,
                [PALETA["panel"]] * 4,
                [PALETA["panel"]] * 4,
                [PALETA["panel"]] * 4,
                colores_ganador,
            ],
            font=dict(color=PALETA["texto"], size=12),
            align="center",
            height=28,
        ),
    ), row=2, col=1)

    fig.update_layout(
        title=(
            "Walk-Forward Cross-Validation (expanding window, 2023–2024)<br>"
            "<sup>LightGBM gana 3/4 folds. Media LGB=421.9 (CV=13.7%) vs Baseline=516.2 (CV=17.1%). "
            "Dentro de la misma distribución, LightGBM supera al baseline por 18%. "
            "El resultado del split principal (2025) refleja parcialmente el declive "
            "interanual de −8.41% — drift fuera de distribución.</sup>"
        ),
        **LAYOUT_BASE,
        height=680,
        legend=dict(x=0.7, y=0.95),
    )
    return fig


# ---------------------------------------------------------------------------
# ENSAMBLADO DEL HTML
# ---------------------------------------------------------------------------
def ensamblar_html(figuras: list, titulo: str = "Metro de Medellín — ML Dashboard") -> str:
    """Genera un HTML autocontenido con todos los gráficos de Plotly."""
    import plotly.io as pio

    partes = [f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>{titulo}</title>
  <style>
    body {{
      background-color: {PALETA["fondo"]};
      color: {PALETA["texto"]};
      font-family: Arial, sans-serif;
      margin: 0; padding: 20px;
    }}
    h1 {{
      text-align: center;
      color: {PALETA["azul"]};
      font-size: 2em;
      margin-bottom: 8px;
    }}
    .subtitulo {{
      text-align: center;
      color: #aaa;
      margin-bottom: 30px;
      font-size: 1.1em;
    }}
    .seccion {{
      margin-bottom: 40px;
      background-color: {PALETA["panel"]};
      border-radius: 10px;
      padding: 15px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.5);
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
  </style>
</head>
<body>
  <h1>Metro de Medellín — Predicción de Afluencia</h1>
  <p class="subtitulo">
    Universidad EAFIT · Git Hug · ML + Big Data + Visualización · 2025<br>
    Train: 2023-01 → 2024-12 &nbsp;|&nbsp; Test: 2025-01 → 2025-12
  </p>
"""]

    seccion_titles = [
        "1. KPIs del modelo ganador",
        "2. Comparación de modelos",
        "3. Real vs Predicho — Top 3 líneas",
        "4. Heatmap de demanda por hora × día",
        "5. Feature Importance — LightGBM",
        "6. Distribución de errores por tipo de línea",
        "7. EDA — Demanda total por línea",
        "8. EDA — Serie temporal del sistema",
        "9. ★ Festivos: MAE normal vs festivo por modelo",
        "10. ★ Festivos: detalle por día festivo",
        "11. Tendencia real: declive 2023–2025",
        "12. Comparación global vs equal-weight por línea",
        "13. Mejora por línea — Baseline vs LGB_PerLinea",
        "14. ★ Walk-forward Cross-Validation",
    ]

    for seccion, fig in zip(seccion_titles, figuras):
        div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if seccion == seccion_titles[0] else False)
        partes.append(f'  <div class="seccion"><h2 style="color:{PALETA["azul"]};margin-top:0">{seccion}</h2>{div}</div>')

    partes.append("</body></html>")
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("[Dashboard] Cargando datos...")
    df_feat    = pd.read_parquet(INPUT_FEATURED)
    df_trusted = pd.read_parquet(INPUT_TRUSTED)
    pred_df    = pd.read_csv(INPUT_PRED, parse_dates=["fecha"])
    imp_df     = pd.read_csv(INPUT_IMP)
    with open(INPUT_METRICS, encoding="utf-8") as f:
        metrics = json.load(f)

    # Asegurar tipo fecha en featured
    df_feat["fecha"] = pd.to_datetime(df_feat["fecha"])

    print("[Dashboard] Construyendo secciones...")

    fig_comp_pl, fig_mejora_pl = build_comparacion_perline(metrics)
    figuras = [
        build_kpis(metrics),
        build_comparacion_modelos(metrics),
        build_real_vs_pred(pred_df, df_trusted),
        build_heatmap(df_feat),
        build_feature_importance(imp_df),
        build_error_dist(pred_df, df_trusted),
        build_eda_total_linea(df_trusted),
        build_eda_serie_temporal(df_trusted),
        build_festivos(metrics, pred_df),
        build_festivos_por_dia(pred_df),
        build_tendencia_anual(df_trusted),
        fig_comp_pl,
        fig_mejora_pl,
        build_walkforward_cv(),
    ]

    print("[Dashboard] Ensamblando HTML...")
    html = ensamblar_html(figuras)

    os.makedirs("data/output", exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(OUTPUT_HTML) / 1_000_000
    print(f"[Dashboard] Guardado: {OUTPUT_HTML}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
