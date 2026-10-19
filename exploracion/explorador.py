"""
Explorador de datos: clon del módulo 'Explorador de datos' del proyecto de
administración (TFG), sobre la tabla LIMPIA.

Mantiene sus apartados (filtros, KPIs, tabla con descarga y evolución del
valor de mercado por jugador) y añade más interacción:
  - Botón para restablecer todos los filtros; el filtro de equipos se
    restringe a las ligas elegidas.
  - Las filas seleccionadas en la tabla pasan directamente a la gráfica de
    evolución.
  - Ficha de jugador: datos principales, evolución de su valor coloreada por
    club y cronología de su carrera.
"""

from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import streamlit as st

from comun import estilo
from comun.banderas import url_bandera
from comun.datos import cargar_tabla, etiquetas_jugadores, perfil_jugadores

# Máximo de filas que se envían a la tabla interactiva (el resto se puede
# descargar): enviar las ~650.000 filas al navegador ralentiza cada interacción.
MAX_FILAS_TABLA = 20_000
CLAVE = "exp_"  # prefijo de las claves de estado de este módulo

# La bandera de la tabla (primera columna) es la de la NACIONALIDAD del
# jugador, no la de la liga que trae la columna `Bandera` de la tabla limpia.
COLUMNAS_TABLA = {
    "bandera_nacionalidad": st.column_config.ImageColumn("", width="small"),
    "nombre": st.column_config.TextColumn("Jugador"),
    "equipo": st.column_config.TextColumn("Equipo"),
    "liga": st.column_config.TextColumn("Liga"),
    "posicion": st.column_config.TextColumn("Posición"),
    "nacionalidad": st.column_config.TextColumn("Nacionalidad"),
    "edad": st.column_config.NumberColumn("Edad", format="%d"),
    "altura_cm": st.column_config.NumberColumn("Altura (cm)", format="%d"),
    "pie": st.column_config.TextColumn("Pie"),
    "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
    "valor_mercado_eur": st.column_config.NumberColumn("Valor de mercado", format="euro"),
    "id_player": st.column_config.NumberColumn("ID", format="%d"),
}


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------
def _restablecer_filtros() -> None:
    for clave in list(st.session_state):
        if clave.startswith(CLAVE + "f_") or clave == CLAVE + "tabla":
            del st.session_state[clave]


def _opciones(serie: pd.Series) -> list[str]:
    return sorted(serie.dropna().unique().tolist())


def _filtros(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown("### Filtros")
        paises = st.multiselect("Nacionalidad", _opciones(df["nacionalidad"]), key=CLAVE + "f_nacionalidad",
                                placeholder="Todas")
        ligas = st.multiselect("Liga", _opciones(df["liga"]), key=CLAVE + "f_liga", placeholder="Todas")
        equipos_disponibles = df.loc[df["liga"].isin(ligas), "equipo"] if ligas else df["equipo"]
        equipos = st.multiselect("Equipo", _opciones(equipos_disponibles), key=CLAVE + "f_equipo",
                                 placeholder="Todos")
        posiciones = st.multiselect("Posición", _opciones(df["posicion"]), key=CLAVE + "f_posicion",
                                    placeholder="Todas")
        nombre = st.text_input("Buscar jugador", key=CLAVE + "f_nombre", placeholder="Nombre o parte del nombre")

        fecha_min, fecha_max = df["fecha"].min().date(), df["fecha"].max().date()
        fechas = st.date_input("Periodo", value=(fecha_min, fecha_max), min_value=fecha_min, max_value=fecha_max,
                               key=CLAVE + "f_fechas", format="DD/MM/YYYY")
        valor_max_m = float(df["valor_mercado_eur"].max()) / 1e6
        rango_m = st.slider("Valor de mercado (M€)", 0.0, valor_max_m, (0.0, valor_max_m), step=0.5,
                            key=CLAVE + "f_valor")
        st.button("↺ Restablecer filtros", on_click=_restablecer_filtros, use_container_width=True)

    mascara = pd.Series(True, index=df.index)
    if paises:
        mascara &= df["nacionalidad"].isin(paises)
    if ligas:
        mascara &= df["liga"].isin(ligas)
    if equipos:
        mascara &= df["equipo"].isin(equipos)
    if posiciones:
        mascara &= df["posicion"].isin(posiciones)
    if nombre:
        mascara &= df["nombre"].str.contains(nombre, case=False, regex=False, na=False)
    if isinstance(fechas, tuple) and len(fechas) == 2:
        mascara &= df["fecha"].between(pd.Timestamp(fechas[0]), pd.Timestamp(fechas[1]))
    if rango_m != (0.0, valor_max_m):
        mascara &= df["valor_mercado_eur"].between(rango_m[0] * 1e6, rango_m[1] * 1e6)
    return df[mascara]


# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------
def _kpis(filtrado: pd.DataFrame) -> None:
    if filtrado.empty:
        estilo.tarjetas_kpi([("Registros", "0", ""), ("Jugadores", "0", ""), ("Valor medio", "—", ""),
                             ("Valor máximo", "—", "")])
        return
    fila_max = filtrado.loc[filtrado["valor_mercado_eur"].idxmax()]
    estilo.tarjetas_kpi([
        ("Registros", estilo.numero(len(filtrado)), f"{filtrado['fecha'].min():%Y} – {filtrado['fecha'].max():%Y}"),
        ("Jugadores", estilo.numero(filtrado["id_player"].nunique()), f"{estilo.numero(filtrado['equipo'].nunique())} equipos"),
        ("Valor medio", estilo.euros(filtrado["valor_mercado_eur"].mean()),
         f"Mediana: {estilo.euros(filtrado['valor_mercado_eur'].median())}"),
        ("Valor máximo", estilo.euros(fila_max["valor_mercado_eur"]), f"{fila_max['nombre']} ({fila_max['fecha']:%Y})"),
    ])


def _tabla(filtrado: pd.DataFrame) -> None:
    mostrar = filtrado.sort_values("fecha", ascending=False)
    if len(mostrar) > MAX_FILAS_TABLA:
        st.caption(
            f"Se muestran los {estilo.numero(MAX_FILAS_TABLA)} registros más recientes de "
            f"{estilo.numero(len(mostrar))}. Afina los filtros o descarga la selección completa."
        )
        mostrar = mostrar.head(MAX_FILAS_TABLA)
    else:
        st.caption("Selecciona una o varias filas para ver la evolución de esos jugadores en la pestaña siguiente.")

    banderas = {n: url_bandera(n) or None for n in mostrar["nacionalidad"].dropna().unique()}
    mostrar = mostrar.assign(bandera_nacionalidad=mostrar["nacionalidad"].map(banderas))
    # Solo las columnas que existan en la tabla limpia cargada (las de edad,
    # altura y pie solo están en las versiones que ya incluyen players.csv).
    columnas = [c for c in COLUMNAS_TABLA if c in mostrar.columns]
    evento = st.dataframe(
        mostrar[columnas],
        column_config=COLUMNAS_TABLA,
        hide_index=True,
        use_container_width=True,
        height=460,
        on_select="rerun",
        selection_mode="multi-row",
        key=CLAVE + "tabla",
    )
    filas = evento.selection.rows if evento and evento.selection else []
    st.session_state[CLAVE + "seleccion"] = mostrar.iloc[filas]["id_player"].unique().tolist() if filas else []

    c1, c2 = st.columns([1, 3])
    if c1.button("⬇️ Preparar descarga CSV", key=CLAVE + "preparar_csv", use_container_width=True):
        st.session_state[CLAVE + "csv"] = filtrado.to_csv(index=False).encode("utf-8")
    if st.session_state.get(CLAVE + "csv"):
        c2.download_button(
            f"Descargar {estilo.numero(len(filtrado))} registros filtrados",
            data=st.session_state[CLAVE + "csv"], file_name="valores_mercado_filtrado.csv", mime="text/csv",
            key=CLAVE + "descargar_csv",
        )


def _etiqueta_jugador(perfiles: pd.DataFrame):
    """`format_func` de los desplegables de jugadores (etiquetas precalculadas)."""
    return etiquetas_jugadores(perfiles, float(len(perfiles))).get


@st.fragment
def _evolucion(filtrado: pd.DataFrame, perfiles: pd.DataFrame) -> None:
    ids_filtrados = filtrado["id_player"].unique()
    seleccion_tabla = [i for i in st.session_state.get(CLAVE + "seleccion", []) if i in set(ids_filtrados)]
    if seleccion_tabla:
        por_defecto = seleccion_tabla
        st.caption("Mostrando los jugadores seleccionados en la tabla.")
    else:
        por_defecto = (
            perfiles.loc[ids_filtrados, "valor_pico"].nlargest(3).index.tolist() if len(ids_filtrados) else []
        )
    opciones = perfiles.loc[ids_filtrados].sort_values("valor_pico", ascending=False).index.tolist()

    c1, c2 = st.columns([4, 1])
    jugadores = c1.multiselect(
        "Jugadores a comparar", opciones, default=por_defecto, format_func=_etiqueta_jugador(perfiles),
        max_selections=10, key=CLAVE + "evo_" + "-".join(map(str, por_defecto)),
        placeholder="Busca uno o varios jugadores",
    )
    log = c2.toggle("Escala logarítmica", key=CLAVE + "evo_log")
    if not jugadores:
        st.info("Elige al menos un jugador para ver la evolución de su valor de mercado.")
        return

    datos = filtrado[filtrado["id_player"].isin(jugadores)].sort_values("fecha")
    datos = datos.assign(jugador=datos["id_player"].map(perfiles["nombre"]))
    fig = px.line(
        datos, x="fecha", y="valor_mercado_eur", color="jugador", markers=True,
        hover_data={"equipo": True, "liga": True, "jugador": False},
        labels={"fecha": "Fecha", "valor_mercado_eur": "Valor de mercado (€)", "jugador": "Jugador"},
    )
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.2), height=480)
    if log:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)


@st.fragment
def _ficha(filtrado: pd.DataFrame, perfiles: pd.DataFrame, df: pd.DataFrame) -> None:
    ids = perfiles.loc[filtrado["id_player"].unique()].sort_values("valor_pico", ascending=False).index.tolist()
    if not ids:
        st.info("No hay jugadores con los filtros actuales.")
        return
    seleccion = st.session_state.get(CLAVE + "seleccion") or []
    indice = ids.index(seleccion[0]) if seleccion and seleccion[0] in ids else 0
    id_player = st.selectbox("Jugador", ids, index=indice, format_func=_etiqueta_jugador(perfiles),
                             key=CLAVE + "ficha_jugador")
    p = perfiles.loc[id_player]
    historial = df[df["id_player"] == id_player].sort_values("fecha")

    c_ficha, c_grafico = st.columns([1, 2])
    bandera_nac = url_bandera(p["nacionalidad"])
    with c_ficha:
        st.markdown(
            f"""<div class="ficha">
            <h3>{html.escape(str(p['nombre']))}</h3>
            <div class="dato">{f'<img src="{bandera_nac}" width="24">' if bandera_nac else ''}{html.escape(str(p['nacionalidad']))}
            · {html.escape(str(p['posicion'] if pd.notna(p['posicion']) else 'Posición desconocida'))}</div>
            <div class="dato">🏟️ <b>{html.escape(str(p['equipo_actual']))}</b>
            {f'<img src="{p["bandera_liga"]}" width="20">' if pd.notna(p['bandera_liga']) else ''}{html.escape(str(p['liga_actual']))}</div>
            <div class="dato">💶 Valor actual: <b>{estilo.euros(p['valor_actual'])}</b></div>
            <div class="dato">🏆 Valor máximo: <b>{estilo.euros(p['valor_pico'])}</b>
            ({p['fecha_pico']:%m/%Y}, {html.escape(str(p['equipo_pico']))})</div>
            <div class="dato">📅 {p['primera_fecha']:%m/%Y} – {p['ultima_fecha']:%m/%Y}
            · {int(p['n_registros'])} valoraciones · {int(p['n_equipos'])} equipos</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c_grafico:
        fig = px.line(historial, x="fecha", y="valor_mercado_eur", markers=True,
                      labels={"fecha": "Fecha", "valor_mercado_eur": "Valor de mercado (€)"})
        fig.update_traces(line_color="#94A3B8")
        fig.add_scatter(x=historial["fecha"], y=historial["valor_mercado_eur"], mode="markers",
                        marker=dict(size=9, color=pd.factorize(historial["equipo"])[0], colorscale="Turbo"),
                        text=historial["equipo"], hovertemplate="%{text}<br>%{x|%d/%m/%Y}: %{y:,.0f} €<extra></extra>",
                        showlegend=False)
        fig.update_layout(height=320, margin=dict(t=40, b=10), title_text="Evolución de su valor de mercado (color = club)")
        st.plotly_chart(fig, use_container_width=True)

    # Cronología de la carrera: un tramo por etapa en cada club.
    cambio_de_club = (historial["equipo"] != historial["equipo"].shift()).fillna(True)
    etapa = cambio_de_club.astype("int64").cumsum()
    carrera = historial.groupby(etapa).agg(
        equipo=("equipo", "first"), pais=("liga", "first"), desde=("fecha", "min"), hasta=("fecha", "max"),
        valoraciones=("fecha", "size"), valor_max=("valor_mercado_eur", "max"),
    ).reset_index(drop=True)
    carrera["hasta_grafico"] = carrera["desde"].shift(-1).fillna(carrera["hasta"] + pd.Timedelta(days=90))
    fig = px.timeline(carrera, x_start="desde", x_end="hasta_grafico", y="equipo", color="pais",
                      hover_data={"valor_max": ":,.0f", "valoraciones": True, "hasta_grafico": False},
                      labels={"equipo": "", "pais": "Liga", "valor_max": "Valor máximo (€)"})
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=120 + 32 * carrera["equipo"].nunique(), margin=dict(t=30, b=10),
                      title="Trayectoria por clubes", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
def render() -> None:
    df = cargar_tabla()
    perfiles = perfil_jugadores(df, float(len(df)))

    estilo.cabecera(
        "Explorador de datos",
        "Filtra, busca y compara el valor de mercado de los jugadores a lo largo del tiempo.",
    )
    filtrado = _filtros(df)
    _kpis(filtrado)
    st.write("")

    tab_tabla, tab_evolucion, tab_ficha = st.tabs(["📋 Tabla de datos", "📈 Evolución del valor", "👤 Ficha de jugador"])
    with tab_tabla:
        if filtrado.empty:
            st.info("No hay registros que cumplan los filtros seleccionados.")
        else:
            _tabla(filtrado)
    with tab_evolucion:
        _evolucion(filtrado, perfiles)
    with tab_ficha:
        _ficha(filtrado, perfiles, df)
