import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Análisis avanzado de respuesta del proceso",
    layout="wide"
)


def normalizar_nombre(col: str) -> str:
    return (
        str(col)
        .strip()
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )


def detectar_columna(candidatas: list[str], columnas: list[str]) -> Optional[str]:
    columnas_norm = {col: normalizar_nombre(col) for col in columnas}
    for candidata in candidatas:
        for col, norm in columnas_norm.items():
            if candidata in norm:
                return col
    return None


def cargar_excel(archivo) -> Tuple[dict, list]:
    xls = pd.ExcelFile(archivo)
    hojas = xls.sheet_names
    data = {hoja: pd.read_excel(archivo, sheet_name=hoja) for hoja in hojas}
    return data, hojas


def convertir_numerico(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columnas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def preparar_datos(
    df: pd.DataFrame,
    col_obj: str,
    col_med: str,
    col_tiempo: Optional[str],
    col_orden: Optional[str]
) -> pd.DataFrame:
    trabajo = df.copy()

    if col_tiempo:
        tiempo_parseado = pd.to_datetime(trabajo[col_tiempo], errors="coerce")
        if tiempo_parseado.notna().sum() > 0:
            trabajo["_tiempo_dt"] = tiempo_parseado
        else:
            trabajo["_tiempo_dt"] = pd.NaT
    else:
        trabajo["_tiempo_dt"] = pd.NaT

    if col_orden:
        trabajo = trabajo.sort_values(by=col_orden).reset_index(drop=True)
    elif trabajo["_tiempo_dt"].notna().sum() > 0:
        trabajo = trabajo.sort_values(by="_tiempo_dt").reset_index(drop=True)
    else:
        trabajo = trabajo.reset_index(drop=True)

    trabajo["_fila"] = np.arange(len(trabajo))

    trabajo = convertir_numerico(trabajo, [col_obj, col_med])

    if col_tiempo and trabajo["_tiempo_dt"].isna().all():
        trabajo[col_tiempo] = pd.to_numeric(trabajo[col_tiempo], errors="coerce")

    trabajo["objetivo_anterior"] = trabajo[col_obj].shift(1)
    trabajo["delta_objetivo"] = trabajo[col_obj] - trabajo["objetivo_anterior"]

    return trabajo


def tiempo_evento(
    df: pd.DataFrame,
    idx_inicio: int,
    idx_fin: Optional[int],
    col_tiempo: Optional[str]
):
    if idx_fin is None:
        return np.nan

    if "_tiempo_dt" in df.columns:
        t0 = df.loc[idx_inicio, "_tiempo_dt"]
        t1 = df.loc[idx_fin, "_tiempo_dt"]
        if pd.notna(t0) and pd.notna(t1):
            return (t1 - t0).total_seconds()

    if col_tiempo and col_tiempo in df.columns:
        t0 = df.loc[idx_inicio, col_tiempo]
        t1 = df.loc[idx_fin, col_tiempo]
        if pd.notna(t0) and pd.notna(t1):
            try:
                return float(t1) - float(t0)
            except Exception:
                return np.nan

    return np.nan


def analizar_eventos_cambio(
    df: pd.DataFrame,
    col_obj: str,
    col_med: str,
    col_tiempo: Optional[str],
    tolerancia_estable: float,
    consecutivos_estable: int,
    cambio_minimo: float
) -> pd.DataFrame:
    eventos = []

    for i in range(1, len(df)):
        prev_obj = df.loc[i - 1, col_obj]
        new_obj = df.loc[i, col_obj]

        if pd.isna(prev_obj) or pd.isna(new_obj):
            continue

        delta = new_obj - prev_obj
        if abs(delta) < cambio_minimo:
            continue

        direccion = "Sube" if delta > 0 else "Baja"
        objetivo_nuevo = new_obj
        objetivo_anterior = prev_obj
        med_inicial = df.loc[i, col_med]
        error_inicial = med_inicial - objetivo_nuevo if pd.notna(med_inicial) else np.nan

        idx_estable = None
        muestras_hasta_estable = np.nan

        for j in range(i, len(df) - consecutivos_estable + 1):
            bloque = df.loc[j:j + consecutivos_estable - 1, col_med]
            if bloque.notna().all():
                dentro = (bloque - objetivo_nuevo).abs() <= tolerancia_estable
                if bool(dentro.all()):
                    idx_estable = j
                    muestras_hasta_estable = j - i
                    break

        tiempo_respuesta = tiempo_evento(df, i, idx_estable, col_tiempo)

        if idx_estable is None:
            serie = df.loc[i:, col_med].dropna()
        else:
            serie = df.loc[i:idx_estable, col_med].dropna()

        overshoot = np.nan
        if not serie.empty:
            if delta > 0:
                overshoot = max(0.0, float((serie - objetivo_nuevo).max()))
            else:
                overshoot = max(0.0, float((objetivo_nuevo - serie).max()))

        eventos.append({
            "idx_evento": i,
            "objetivo_anterior": objetivo_anterior,
            "objetivo_nuevo": objetivo_nuevo,
            "delta_objetivo": delta,
            "direccion": direccion,
            "diametro_medido_inicial": med_inicial,
            "error_inicial": error_inicial,
            "idx_estable": idx_estable,
            "muestras_hasta_estable": muestras_hasta_estable,
            "tiempo_respuesta_seg": tiempo_respuesta,
            "overshoot_mm": overshoot,
            "transicion": f"{objetivo_anterior:.4f} -> {objetivo_nuevo:.4f}",
        })

    return pd.DataFrame(eventos)


def resumen_eventos(df_eventos: pd.DataFrame) -> pd.DataFrame:
    if df_eventos.empty:
        return pd.DataFrame(columns=[
            "transicion", "eventos", "tiempo_promedio_seg", "tiempo_mediana_seg",
            "muestras_promedio", "overshoot_promedio_mm"
        ])

    return (
        df_eventos.groupby("transicion", dropna=False)
        .agg(
            eventos=("transicion", "count"),
            tiempo_promedio_seg=("tiempo_respuesta_seg", "mean"),
            tiempo_mediana_seg=("tiempo_respuesta_seg", "median"),
            muestras_promedio=("muestras_hasta_estable", "mean"),
            overshoot_promedio_mm=("overshoot_mm", "mean")
        )
        .reset_index()
    )


def filtrar_transiciones_especificas(
    df_eventos: pd.DataFrame,
    valor_bajo: float,
    valor_alto: float,
    atol: float = 1e-6
) -> pd.DataFrame:
    if df_eventos.empty:
        return df_eventos.copy()

    cond1 = (
        np.isclose(df_eventos["objetivo_anterior"], valor_bajo, atol=atol) &
        np.isclose(df_eventos["objetivo_nuevo"], valor_alto, atol=atol)
    )
    cond2 = (
        np.isclose(df_eventos["objetivo_anterior"], valor_alto, atol=atol) &
        np.isclose(df_eventos["objetivo_nuevo"], valor_bajo, atol=atol)
    )
    return df_eventos[cond1 | cond2].copy()


def exportar_excel(
    df_datos: pd.DataFrame,
    df_eventos: pd.DataFrame,
    df_resumen: pd.DataFrame,
    df_especificos: pd.DataFrame
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_datos.to_excel(writer, sheet_name="datos_procesados", index=False)
        df_eventos.to_excel(writer, sheet_name="eventos_cambio", index=False)
        df_resumen.to_excel(writer, sheet_name="resumen_eventos", index=False)
        df_especificos.to_excel(writer, sheet_name="eventos_0.2_0.5", index=False)
    return output.getvalue()


st.title("Análisis avanzado de tiempo de respuesta del proceso")

with st.sidebar:
    st.header("Configuración")

    archivo = st.file_uploader("Sube tu Excel", type=["xlsx", "xls"])

    valor_bajo = st.number_input(
        "Valor bajo de transición",
        value=0.2,
        step=0.1,
        format="%.4f"
    )
    valor_alto = st.number_input(
        "Valor alto de transición",
        value=0.5,
        step=0.1,
        format="%.4f"
    )

    cambio_minimo = st.number_input(
        "Cambio mínimo para detectar evento (mm)",
        value=0.05,
        step=0.01,
        format="%.4f"
    )
    tolerancia_estable = st.number_input(
        "Tolerancia para considerar estable (± mm)",
        value=0.02,
        step=0.005,
        format="%.4f"
    )
    consecutivos_estable = st.number_input(
        "Puntos consecutivos dentro de tolerancia",
        min_value=1,
        value=3,
        step=1
    )

if not archivo:
    st.info("Sube un archivo Excel para comenzar.")
    st.stop()

try:
    data_hojas, hojas = cargar_excel(archivo)
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

hoja = st.selectbox("Selecciona la hoja", hojas)
df_original = data_hojas[hoja].copy()

if df_original.empty:
    st.warning("La hoja seleccionada está vacía.")
    st.stop()

st.subheader("Vista previa")
st.dataframe(df_original.head(20), use_container_width=True)

columnas = df_original.columns.tolist()

sug_obj = detectar_columna(["diametro objetivo", "objetivo", "target"], columnas)
sug_med = detectar_columna(["diametro medido", "medido", "real", "measured"], columnas)
sug_tiempo = detectar_columna(["tiempo", "seg", "timestamp", "fecha", "hora"], columnas)
sug_orden = detectar_columna(["orden", "secuencia", "id", "index"], columnas)

c1, c2, c3, c4 = st.columns(4)

with c1:
    col_obj = st.selectbox(
        "Columna diámetro objetivo",
        columnas,
        index=columnas.index(sug_obj) if sug_obj in columnas else 0
    )

with c2:
    col_med = st.selectbox(
        "Columna diámetro medido",
        columnas,
        index=columnas.index(sug_med) if sug_med in columnas else 0
    )

with c3:
    opciones_tiempo = ["(sin columna de tiempo)"] + columnas
    idx_tiempo = opciones_tiempo.index(sug_tiempo) if sug_tiempo in columnas else 0
    col_tiempo_sel = st.selectbox("Columna tiempo", opciones_tiempo, index=idx_tiempo)
    col_tiempo = None if col_tiempo_sel == "(sin columna de tiempo)" else col_tiempo_sel

with c4:
    opciones_orden = ["(sin columna de orden)"] + columnas
    idx_orden = opciones_orden.index(sug_orden) if sug_orden in columnas else 0
    col_orden_sel = st.selectbox("Columna orden", opciones_orden, index=idx_orden)
    col_orden = None if col_orden_sel == "(sin columna de orden)" else col_orden_sel

df = preparar_datos(
    df_original,
    col_obj=col_obj,
    col_med=col_med,
    col_tiempo=col_tiempo,
    col_orden=col_orden
)

df = df.dropna(subset=[col_obj, col_med]).copy()

if df.empty:
    st.warning("No hay datos numéricos válidos en diámetro objetivo y diámetro medido.")
    st.stop()

eventos = analizar_eventos_cambio(
    df=df,
    col_obj=col_obj,
    col_med=col_med,
    col_tiempo=col_tiempo,
    tolerancia_estable=tolerancia_estable,
    consecutivos_estable=int(consecutivos_estable),
    cambio_minimo=cambio_minimo
)

resumen = resumen_eventos(eventos)
eventos_especificos = filtrar_transiciones_especificas(
    eventos,
    valor_bajo=valor_bajo,
    valor_alto=valor_alto
)

st.subheader("Serie temporal del proceso")

if col_tiempo and "_tiempo_dt" in df.columns and df["_tiempo_dt"].notna().sum() > 0:
    x_line = "_tiempo_dt"
elif col_tiempo:
    x_line = col_tiempo
else:
    x_line = "_fila"

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=df[x_line],
    y=df[col_med],
    mode="lines+markers",
    name="Diámetro medido"
))
fig_line.add_trace(go.Scatter(
    x=df[x_line],
    y=df[col_obj],
    mode="lines",
    name="Diámetro objetivo"
))
fig_line.update_layout(
    xaxis_title="Tiempo",
    yaxis_title="Diámetro"
)
st.plotly_chart(fig_line, use_container_width=True)

st.subheader("Respuesta del proceso: tiempo en X y diámetro medido en Y")

if col_tiempo:
    x_time = "_tiempo_dt" if df["_tiempo_dt"].notna().sum() > 0 else col_tiempo

    fig_scatter = px.scatter(
        df,
        x=x_time,
        y=col_med,
        color=col_obj,
        title="Diámetro medido en función del tiempo",
        hover_data=[col_obj]
    )
    fig_scatter.update_layout(
        xaxis_title="Tiempo",
        yaxis_title="Diámetro medido"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Para esta gráfica necesitas seleccionar una columna de tiempo.")

st.subheader("Resumen general de eventos")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Eventos detectados", int(len(eventos)))
m2.metric("Transiciones 0.2 ↔ 0.5", int(len(eventos_especificos)))
m3.metric(
    "Tiempo promedio de respuesta (seg)",
    f"{eventos['tiempo_respuesta_seg'].mean():.2f}"
    if not eventos.empty and eventos["tiempo_respuesta_seg"].notna().any()
    else "N/D"
)
m4.metric(
    "Overshoot promedio (mm)",
    f"{eventos['overshoot_mm'].mean():.4f}"
    if not eventos.empty and eventos["overshoot_mm"].notna().any()
    else "N/D"
)

st.dataframe(resumen, use_container_width=True)

st.subheader(f"Transiciones específicas {valor_bajo:.4f} ↔ {valor_alto:.4f}")

if eventos_especificos.empty:
    st.warning("No se encontraron esas transiciones específicas.")
else:
    resumen_especifico = (
        eventos_especificos.groupby("transicion")
        .agg(
            eventos=("transicion", "count"),
            tiempo_promedio_seg=("tiempo_respuesta_seg", "mean"),
            tiempo_mediana_seg=("tiempo_respuesta_seg", "median"),
            muestras_promedio=("muestras_hasta_estable", "mean"),
            overshoot_promedio_mm=("overshoot_mm", "mean"),
        )
        .reset_index()
    )

    st.dataframe(resumen_especifico, use_container_width=True)

    c5, c6 = st.columns(2)
    trans_up = eventos_especificos[
        np.isclose(eventos_especificos["objetivo_anterior"], valor_bajo) &
        np.isclose(eventos_especificos["objetivo_nuevo"], valor_alto)
    ]
    trans_down = eventos_especificos[
        np.isclose(eventos_especificos["objetivo_anterior"], valor_alto) &
        np.isclose(eventos_especificos["objetivo_nuevo"], valor_bajo)
    ]

    with c5:
        st.metric(
            f"Tiempo promedio {valor_bajo:.1f} -> {valor_alto:.1f}",
            f"{trans_up['tiempo_respuesta_seg'].mean():.2f} s"
            if not trans_up.empty and trans_up["tiempo_respuesta_seg"].notna().any()
            else "N/D"
        )

    with c6:
        st.metric(
            f"Tiempo promedio {valor_alto:.1f} -> {valor_bajo:.1f}",
            f"{trans_down['tiempo_respuesta_seg'].mean():.2f} s"
            if not trans_down.empty and trans_down["tiempo_respuesta_seg"].notna().any()
            else "N/D"
        )

    fig_box = px.box(
        eventos_especificos,
        x="transicion",
        y="tiempo_respuesta_seg",
        points="all",
        title="Distribución del tiempo de respuesta por transición"
    )
    fig_box.update_layout(
        xaxis_title="Transición",
        yaxis_title="Tiempo de respuesta (seg)"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    fig_over = px.bar(
        eventos_especificos,
        x=eventos_especificos.index.astype(str),
        y="overshoot_mm",
        color="transicion",
        title="Overshoot por evento"
    )
    fig_over.update_layout(
        xaxis_title="Evento",
        yaxis_title="Overshoot (mm)"
    )
    st.plotly_chart(fig_over, use_container_width=True)

    st.write("Detalle de eventos detectados")
    st.dataframe(eventos_especificos, use_container_width=True)

st.subheader("Todos los eventos detectados")
st.dataframe(eventos, use_container_width=True)

excel_out = exportar_excel(df, eventos, resumen, eventos_especificos)

st.download_button(
    "Descargar resultados en Excel",
    data=excel_out,
    file_name="analisis_avanzado_respuesta.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
