import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Análisis de tiempos de respuesta vs diámetro objetivo",
    layout="wide"
)


# ---------------------------
# Utilidades
# ---------------------------
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


def clasificar_cambio(delta: float, umbral: float) -> str:
    if pd.isna(delta):
        return "Sin dato previo"
    if delta >= umbral:
        return "Aumento"
    if delta <= -umbral:
        return "Disminución"
    return "Estable"


def cargar_excel(archivo) -> Tuple[dict, list]:
    xls = pd.ExcelFile(archivo)
    hojas = xls.sheet_names
    data = {hoja: pd.read_excel(archivo, sheet_name=hoja) for hoja in hojas}
    return data, hojas


def convertir_a_numerico(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columnas:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def preparar_datos(
    df: pd.DataFrame,
    col_diametro: str,
    col_tiempo: str,
    col_orden: Optional[str],
    umbral_mm: float
) -> pd.DataFrame:
    trabajo = df.copy()

    if col_orden:
        trabajo = trabajo.sort_values(by=col_orden).reset_index(drop=True)
    else:
        trabajo = trabajo.reset_index(drop=True)
        trabajo["_orden_fila"] = np.arange(len(trabajo))

    columnas_numericas = [col_diametro, col_tiempo]
    if col_orden:
        columnas_numericas.append(col_orden)

    trabajo = convertir_a_numerico(trabajo, columnas_numericas)

    trabajo["diametro_anterior"] = trabajo[col_diametro].shift(1)
    trabajo["delta_diametro"] = trabajo[col_diametro] - trabajo["diametro_anterior"]
    trabajo["clasificacion"] = trabajo["delta_diametro"].apply(
        lambda x: clasificar_cambio(x, umbral_mm)
    )
    trabajo["delta_abs"] = trabajo["delta_diametro"].abs()

    return trabajo


def resumen_estadistico(df: pd.DataFrame, col_tiempo: str) -> pd.DataFrame:
    resumen = (
        df.groupby("clasificacion", dropna=False)[col_tiempo]
        .agg(
            conteo="count",
            promedio="mean",
            mediana="median",
            minimo="min",
            maximo="max",
            desviacion_std="std"
        )
        .reset_index()
    )
    return resumen


def generar_excel_descarga(df_detalle: pd.DataFrame, df_resumen: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_detalle.to_excel(writer, sheet_name="detalle_analisis", index=False)
        df_resumen.to_excel(writer, sheet_name="resumen", index=False)
    return output.getvalue()


# ---------------------------
# UI
# ---------------------------
st.title("Análisis de tiempos de respuesta vs diámetro objetivo")
st.markdown(
    """
Esta aplicación analiza cómo cambian los **tiempos de respuesta**
cuando el **diámetro objetivo aumenta o disminuye** según un rango configurable.
"""
)

with st.sidebar:
    st.header("Configuración")
    umbral_mm = st.number_input(
        "Rango de cambio del diámetro (mm)",
        min_value=0.0001,
        value=0.1,
        step=0.01,
        format="%.4f",
        help="Cambios mayores o iguales a este valor se clasifican como aumento o disminución."
    )

    archivo = st.file_uploader(
        "Sube tu archivo Excel",
        type=["xlsx", "xls"]
    )

if not archivo:
    st.info("Sube un archivo Excel para comenzar.")
    st.stop()

# ---------------------------
# Lectura del Excel
# ---------------------------
try:
    data_hojas, hojas = cargar_excel(archivo)
except Exception as e:
    st.error(f"No se pudo leer el archivo Excel: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    hoja_seleccionada = st.selectbox("Selecciona la hoja", hojas)

df_original = data_hojas[hoja_seleccionada].copy()

if df_original.empty:
    st.warning("La hoja seleccionada no contiene datos.")
    st.stop()

st.subheader("Vista previa del archivo")
st.dataframe(df_original.head(20), use_container_width=True)

columnas = df_original.columns.tolist()

# Autodetección básica
sugerida_diametro = detectar_columna(
    ["diametro objetivo", "diametro", "target diameter", "objetivo"],
    columnas
)
sugerida_tiempo = detectar_columna(
    ["tiempo de respuesta", "response time", "tiempo", "respuesta"],
    columnas
)
sugerida_orden = detectar_columna(
    ["timestamp", "fecha", "hora", "datetime", "orden", "secuencia", "id"],
    columnas
)

st.subheader("Mapeo de columnas")

c1, c2, c3 = st.columns(3)
with c1:
    col_diametro = st.selectbox(
        "Columna de diámetro objetivo",
        options=columnas,
        index=columnas.index(sugerida_diametro) if sugerida_diametro in columnas else 0
    )
with c2:
    col_tiempo = st.selectbox(
        "Columna de tiempo de respuesta",
        options=columnas,
        index=columnas.index(sugerida_tiempo) if sugerida_tiempo in columnas else min(1, len(columnas) - 1)
    )
with c3:
    opciones_orden = ["(usar orden actual de filas)"] + columnas
    idx_orden = 0
    if sugerida_orden in columnas:
        idx_orden = opciones_orden.index(sugerida_orden)
    col_orden_seleccion = st.selectbox("Columna de orden/tiempo", options=opciones_orden, index=idx_orden)

col_orden = None if col_orden_seleccion == "(usar orden actual de filas)" else col_orden_seleccion

# ---------------------------
# Procesamiento
# ---------------------------
try:
    df_analisis = preparar_datos(
        df=df_original,
        col_diametro=col_diametro,
        col_tiempo=col_tiempo,
        col_orden=col_orden,
        umbral_mm=umbral_mm
    )
except Exception as e:
    st.error(f"Error al procesar los datos: {e}")
    st.stop()

df_validos = df_analisis.dropna(subset=[col_diametro, col_tiempo]).copy()

if df_validos.empty:
    st.warning("No hay datos numéricos válidos en las columnas seleccionadas.")
    st.stop()

resumen = resumen_estadistico(df_validos, col_tiempo)

# ---------------------------
# Métricas
# ---------------------------
st.subheader("Resumen general")

aumentos = (df_validos["clasificacion"] == "Aumento").sum()
disminuciones = (df_validos["clasificacion"] == "Disminución").sum()
estables = (df_validos["clasificacion"] == "Estable").sum()

prom_aum = df_validos.loc[df_validos["clasificacion"] == "Aumento", col_tiempo].mean()
prom_dis = df_validos.loc[df_validos["clasificacion"] == "Disminución", col_tiempo].mean()
prom_est = df_validos.loc[df_validos["clasificacion"] == "Estable", col_tiempo].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Aumentos", int(aumentos))
m2.metric("Disminuciones", int(disminuciones))
m3.metric("Estables", int(estables))
m4.metric("Umbral actual (mm)", f"{umbral_mm:.4f}")

m5, m6, m7 = st.columns(3)
m5.metric("Promedio tiempo - Aumento", f"{prom_aum:.4f}" if pd.notna(prom_aum) else "N/D")
m6.metric("Promedio tiempo - Disminución", f"{prom_dis:.4f}" if pd.notna(prom_dis) else "N/D")
m7.metric("Promedio tiempo - Estable", f"{prom_est:.4f}" if pd.notna(prom_est) else "N/D")

st.subheader("Tabla resumen")
st.dataframe(resumen, use_container_width=True)

# ---------------------------
# Gráficos
# ---------------------------
st.subheader("Gráficos")

tab1, tab2, tab3, tab4 = st.tabs([
    "Serie temporal / secuencia",
    "Distribución por clasificación",
    "Boxplot comparativo",
    "Detalle analizado"
])

with tab1:
    eje_x = col_orden if col_orden else (df_validos.index + 1)

    fig_linea = px.line(
        df_validos,
        x=eje_x,
        y=col_tiempo,
        color="clasificacion",
        markers=True,
        title="Tiempo de respuesta a lo largo de la secuencia"
    )
    st.plotly_chart(fig_linea, use_container_width=True)

    fig_diametro = px.line(
        df_validos,
        x=eje_x,
        y=col_diametro,
        markers=True,
        title="Diámetro objetivo a lo largo de la secuencia"
    )
    st.plotly_chart(fig_diametro, use_container_width=True)

with tab2:
    fig_barras = px.bar(
        resumen,
        x="clasificacion",
        y="promedio",
        title="Promedio del tiempo de respuesta por clasificación",
        text_auto=".3f"
    )
    st.plotly_chart(fig_barras, use_container_width=True)

    fig_hist = px.histogram(
        df_validos,
        x=col_tiempo,
        color="clasificacion",
        barmode="overlay",
        nbins=30,
        title="Distribución de tiempos de respuesta"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    fig_box = px.box(
        df_validos[df_validos["clasificacion"].isin(["Aumento", "Disminución", "Estable"])],
        x="clasificacion",
        y=col_tiempo,
        points="all",
        title="Comparación de tiempos de respuesta por clasificación"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    fig_scatter = px.scatter(
        df_validos,
        x="delta_diametro",
        y=col_tiempo,
        color="clasificacion",
        hover_data=[col_diametro],
        title="Relación entre cambio de diámetro y tiempo de respuesta"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab4:
    columnas_mostrar = [
        col_diametro,
        "diametro_anterior",
        "delta_diametro",
        "delta_abs",
        "clasificacion",
        col_tiempo,
    ]
    if col_orden:
        columnas_mostrar = [col_orden] + columnas_mostrar

    st.dataframe(
        df_validos[columnas_mostrar],
        use_container_width=True
    )

# ---------------------------
# Descarga
# ---------------------------
st.subheader("Descargar resultados")

excel_salida = generar_excel_descarga(
    df_detalle=df_validos,
    df_resumen=resumen
)

st.download_button(
    label="Descargar análisis en Excel",
    data=excel_salida,
    file_name="analisis_tiempos_respuesta.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ---------------------------
# Explicación
# ---------------------------
with st.expander("Cómo interpreta la app los cambios de diámetro"):
    st.write(
        f"""
- **Aumento**: cuando el cambio de diámetro entre una fila y la anterior es mayor o igual a **{umbral_mm:.4f} mm**
- **Disminución**: cuando el cambio es menor o igual a **-{umbral_mm:.4f} mm**
- **Estable**: cuando el cambio queda entre esos dos límites
- **Sin dato previo**: la primera fila no tiene una fila anterior para comparar
"""
    )
