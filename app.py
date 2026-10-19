"""
Portal de usuario — Valores de mercado de futbolistas.

Aplicación Streamlit de CONSULTA para usuarios/clientes. Trabaja siempre
sobre la tabla limpia de `datos/`, que generan los administradores en el
proyecto TFG (pestaña "Compilación") y se copia aquí a mano.

Ejecutar desde esta carpeta:

    streamlit run app.py

Para añadir un módulo nuevo: crear su carpeta con una función `render()` y
registrarla en `PAGINAS` (más abajo).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import exploracion
import fichajes
import mejores_fichajes
import plantillas
import prediccion
import visualizacion
from comun import estilo, navegacion
from comun.banderas import url_bandera
from comun.datos import cargar_tabla, fecha_actualizacion, perfil_jugadores

st.set_page_config(
    page_title="Mercagoal · Valores de mercado",
    page_icon=str(estilo.DIR_ASSETS / "icono.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
estilo.aplicar_estilo()


# ---------------------------------------------------------------------------
# Página de inicio
# ---------------------------------------------------------------------------
@st.cache_data
def _resumen_inicio(_df: pd.DataFrame, n: int):
    por_anio = (
        _df.assign(anio=_df["fecha"].dt.year).groupby("anio")
        .agg(valor_medio=("valor_mercado_eur", "mean"), jugadores=("id_player", "nunique"))
        .reset_index()
    )
    por_pais = (
        _df.groupby("liga").agg(valor_total=("valor_mercado_eur", "sum"), registros=("id_player", "size"))
        .nlargest(10, "valor_total").reset_index()
    )
    return por_anio, por_pais


def inicio() -> None:
    df = cargar_tabla()
    perfiles = perfil_jugadores(df, float(len(df)))
    estilo.cabecera(
        "Valores de mercado de futbolistas",
        "Consulta y análisis del histórico de valoraciones de más de 40.000 jugadores.",
    )

    estilo.tarjetas_kpi([
        ("Valoraciones", estilo.numero(len(df)), f"{df['fecha'].min():%Y} – {df['fecha'].max():%Y}"),
        ("Jugadores", estilo.numero(df["id_player"].nunique()), f"{estilo.numero(df['nacionalidad'].nunique())} nacionalidades"),
        ("Equipos", estilo.numero(df["equipo"].nunique()), f"{df['liga'].nunique()} ligas"),
        ("Valor medio", estilo.euros(df["valor_mercado_eur"].mean()), f"Mediana: {estilo.euros(df['valor_mercado_eur'].median())}"),
    ])
    st.write("")

    por_anio, por_pais = _resumen_inicio(df, len(df))
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        fig = px.area(por_anio, x="anio", y="valor_medio",
                      labels={"anio": "Año", "valor_medio": "Valor medio (€)"},
                      title="Valor de mercado medio por año")
        fig.update_layout(height=360, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(por_pais.sort_values("valor_total"), x="valor_total", y="liga", orientation="h",
                     labels={"valor_total": "Valor acumulado (€)", "liga": ""},
                     title="Ligas con más valor acumulado")
        fig.update_layout(height=360, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🏆 Jugadores con mayor valor de mercado alcanzado")
    top = perfiles.nlargest(10, "valor_pico").reset_index()
    # Bandera de la NACIONALIDAD del jugador (no la de la liga de su club).
    top["bandera_nacionalidad"] = top["nacionalidad"].map(lambda n: url_bandera(n) or None)
    st.dataframe(
        top[["bandera_nacionalidad", "nombre", "nacionalidad", "posicion", "edad_pico", "altura_cm", "pie",
             "equipo_pico", "valor_pico", "fecha_pico", "valor_actual"]],
        hide_index=True, use_container_width=True,
        column_config={
            "bandera_nacionalidad": st.column_config.ImageColumn("", width="small"),
            "nombre": "Jugador", "nacionalidad": "Nacionalidad", "posicion": "Posición",
            "edad_pico": st.column_config.NumberColumn("Edad en su máximo", format="%d"),
            "altura_cm": st.column_config.NumberColumn("Altura (cm)", format="%d"),
            "pie": "Pie",
            "equipo_pico": "Equipo en su máximo",
            "valor_pico": st.column_config.ProgressColumn(
                "Valor máximo", format="%.0f €", min_value=0, max_value=float(top["valor_pico"].max())),
            "fecha_pico": st.column_config.DateColumn("Fecha del máximo", format="MM/YYYY"),
            "valor_actual": st.column_config.NumberColumn("Último valor", format="euro"),
        },
    )

    st.markdown("#### Módulos")
    m1, m2, m3 = st.columns(3)
    with m1.container(border=True):
        st.markdown("**🔎 Explorador de datos**  \nFiltra y busca registros, compara la evolución de varios "
                    "jugadores y consulta la ficha y trayectoria de cada uno.")
        st.page_link(PAGINA_EXPLORADOR, label="Abrir el explorador", icon=":material/arrow_forward:")
    with m2.container(border=True):
        st.markdown("**📊 Visualización de datos**  \nDistribuciones, rankings, series temporales y la red de "
                    "traspasos entre ligas y equipos.")
        st.page_link(PAGINA_VISUALIZACION, label="Abrir la visualización", icon=":material/arrow_forward:")
    with m3.container(border=True):
        st.markdown("**🔮 Predicción de valores de mercado**  \nSimulación de Monte Carlo de la carrera de un "
                    "jugador: escenarios, probabilidad de que suba y riesgo a varios años vista.")
        st.page_link(PAGINA_PREDICCION, label="Abrir la predicción", icon=":material/arrow_forward:")
    m4, m5, m6 = st.columns(3)
    with m4.container(border=True):
        st.markdown("**🤝 Medidor de fichajes**  \nOferta máxima por un jugador para ganar dinero al revenderlo "
                    "con la probabilidad que elijas, según el historial de traspasos.")
        st.page_link(PAGINA_FICHAJES, label="Abrir el medidor", icon=":material/arrow_forward:")
    with m5.container(border=True):
        st.markdown("**🏆 Mejores fichajes**  \nRanking de los jugadores más rentables que puede fichar un equipo "
                    "por posición, presupuesto y categorías.")
        st.page_link(PAGINA_MEJORES, label="Abrir el ranking", icon=":material/arrow_forward:")
    with m6.container(border=True):
        st.markdown("**💼 Gestión de plantillas**  \nLa plantilla de cada club como una cartera de inversión: "
                    "reparto del valor y diversificación según la frontera eficiente de Markowitz.")
        st.page_link(PAGINA_PLANTILLAS, label="Abrir la gestión de plantillas", icon=":material/arrow_forward:")


# ---------------------------------------------------------------------------
# Navegación
# ---------------------------------------------------------------------------
PAGINA_INICIO = st.Page(inicio, title="Inicio", icon=":material/home:", default=True)
PAGINA_EXPLORADOR = st.Page(exploracion.render, title="Explorador de datos", icon=":material/travel_explore:",
                            url_path="explorador")
PAGINA_VISUALIZACION = st.Page(visualizacion.render, title="Visualización de datos", icon=":material/insights:",
                               url_path="visualizacion")

PAGINA_PREDICCION = st.Page(prediccion.render, title="Predicción de valores de mercado", icon=":material/query_stats:",
                            url_path="prediccion")

PAGINA_FICHAJES = st.Page(fichajes.render, title="Medidor de fichajes", icon=":material/handshake:",
                          url_path="fichajes")

PAGINA_MEJORES = st.Page(mejores_fichajes.render, title="Mejores fichajes", icon=":material/trophy:",
                         url_path="mejores-fichajes")

PAGINA_PLANTILLAS = st.Page(plantillas.render, title="Gestión de plantillas", icon=":material/pie_chart:",
                            url_path="plantillas")

# Registro para que los módulos puedan enlazarse entre sí (st.switch_page).
navegacion.PAGINAS.update(fichajes=PAGINA_FICHAJES, mejores_fichajes=PAGINA_MEJORES)

PAGINAS = {
    "": [PAGINA_INICIO],
    "Módulos": [PAGINA_EXPLORADOR, PAGINA_VISUALIZACION, PAGINA_PREDICCION, PAGINA_FICHAJES, PAGINA_MEJORES,
                PAGINA_PLANTILLAS],
}

with st.sidebar:
    st.caption("ANÁLISIS FINANCIERO DEL FÚTBOL")
    actualizado = fecha_actualizacion()
    if actualizado:
        st.caption(f"Datos actualizados el {actualizado:%d/%m/%Y}")

st.logo(str(estilo.DIR_ASSETS / "logo.png"), size="large", icon_image=str(estilo.DIR_ASSETS / "icono.png"))
st.navigation(PAGINAS).run()
