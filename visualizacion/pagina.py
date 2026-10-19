"""
Visualización de datos: clon del módulo 'Visualización de datos' del
proyecto de administración (TFG), sobre la tabla LIMPIA.

Mismos tipos de gráfica (boxplot, densidad, barras de frecuencia y de
agregación, serie temporal y red de traspasos), con los controles a la
izquierda y el resultado a la derecha. Cada gráfica vive en un
`st.fragment`: al cambiar un control solo se recalcula esa gráfica, no toda
la página, y las figuras pesadas se cachean.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import estilo
from comun.banderas import banderas_en_eje_y
from comun.datos import cargar_tabla

from . import graficas as gr
from . import red_de_traspasos as rt

CLAVE = "vis_"
# Variables cuyos valores son países: sus gráficos de barras llevan bandera.
COLUMNAS_CON_BANDERA = {"liga", "nacionalidad"}
ETIQUETAS = {
    "valor_mercado_eur": "Valor de mercado (€)", "liga": "Liga", "nacionalidad": "Nacionalidad",
    "posicion": "Posición", "equipo": "Equipo", "fecha": "Fecha", "edad": "Edad",
    "altura_cm": "Altura (cm)", "pie": "Pie dominante",
}
# Variable numérica que aparece seleccionada por defecto en todas las gráficas.
VARIABLE_POR_DEFECTO = "valor_mercado_eur"


def _indice_por_defecto(columnas: list[str]) -> int:
    return columnas.index(VARIABLE_POR_DEFECTO) if VARIABLE_POR_DEFECTO in columnas else 0
TIPOS = {
    "Boxplot": ("📦", "Resumen de mediana, cuartiles y valores atípicos de una variable numérica."),
    "Densidad (KDE)": ("〰️", "Curva suavizada (estimación de densidad) de la distribución de una variable."),
    "Barras — frecuencia": ("📊", "Nº de registros por cada valor de una variable categórica."),
    "Barras — agregación": ("📶", "Media, mediana, suma... de una variable numérica, por categoría."),
    "Serie temporal": ("📈", "Evolución en el tiempo de una variable numérica, agregada por periodo."),
    "Red de traspasos": ("🕸️", "Red de traspasos de jugadores entre ligas o equipos: cada flecha va del origen al destino."),
}


def _etiqueta(columna: str | None) -> str:
    return ETIQUETAS.get(columna, columna) if columna else "Ninguna"


def _filtrar_top(df: pd.DataFrame, columna: str | None, n: int) -> pd.DataFrame:
    if not columna:
        return df
    return df[df[columna].isin(gr.top_n_categorias(df, columna, n))]


# ---------------------------------------------------------------------------
# Figuras cacheadas (el DataFrame, con "_", no se hashea: es siempre la
# misma tabla limpia, cargada una vez)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calculando la gráfica...")
def _fig_boxplot(_df, col_num, col_cat, top_n, log, atipicos) -> go.Figure:
    fig = gr.boxplot(_filtrar_top(_df, col_cat, top_n), col_num, columna_cat=col_cat, escala_log_y=log,
                     titulo=f"{_etiqueta(col_num)}" + (f" por {_etiqueta(col_cat).lower()}" if col_cat else ""))
    if not atipicos:
        fig.update_traces(boxpoints=False)
    return fig


@st.cache_data(show_spinner="Estimando la densidad...")
def _fig_densidad(_df, col_num, col_cat, top_n, log) -> go.Figure:
    return gr.densidad(_filtrar_top(_df, col_cat, top_n), col_num, color_por=col_cat, escala_log_x=log,
                       titulo=f"Densidad de {_etiqueta(col_num).lower()}"
                       + (f" por {_etiqueta(col_cat).lower()}" if col_cat else ""))


@st.cache_data
def _fig_serie(_df, col_fecha, col_num, col_cat, top_n, frecuencia, agregacion) -> go.Figure:
    return gr.serie_temporal(_filtrar_top(_df, col_cat, top_n), col_fecha, col_num, columna_cat=col_cat,
                             frecuencia=frecuencia, agregacion=agregacion)


def _pintar(fig: go.Figure, altura: int = 560) -> None:
    # Barras horizontales con la cifra al final: margen derecho y sin
    # recortar el texto, para que la cifra de la barra más larga se vea entera.
    es_barras = any(getattr(traza, "type", "") == "bar" for traza in fig.data)
    if es_barras:
        fig.update_traces(cliponaxis=False)
    fig.update_layout(height=altura, margin=dict(t=60, l=10, r=70 if es_barras else 10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Un fragmento por tipo de gráfica
# ---------------------------------------------------------------------------
@st.fragment
def _boxplot(df, cols_num, cols_cat) -> None:
    c_ctrl, c_graf = st.columns([1, 3], gap="large")
    with c_ctrl:
        col_num = st.selectbox("Variable numérica", cols_num, index=_indice_por_defecto(cols_num), format_func=_etiqueta, key=CLAVE + "box_num")
        col_cat = st.selectbox("Comparar por", [None] + cols_cat, format_func=_etiqueta, key=CLAVE + "box_cat")
        top_n = st.slider("Nº de categorías", 2, 30, 10, key=CLAVE + "box_n", disabled=not col_cat)
        log = st.toggle("Escala logarítmica", value=True, key=CLAVE + "box_log")
        atipicos = st.toggle("Mostrar valores atípicos", value=False, key=CLAVE + "box_atip",
                             help="Dibujar los miles de puntos atípicos hace la gráfica más lenta.")
    with c_graf:
        fig = _fig_boxplot(df, col_num, col_cat, top_n, log, atipicos)
        fig.update_yaxes(title_text=_etiqueta(col_num))
        fig.update_xaxes(title_text=_etiqueta(col_cat) if col_cat else "")
        _pintar(fig)


@st.fragment
def _densidad(df, cols_num, cols_cat) -> None:
    c_ctrl, c_graf = st.columns([1, 3], gap="large")
    with c_ctrl:
        col_num = st.selectbox("Variable numérica", cols_num, index=_indice_por_defecto(cols_num), format_func=_etiqueta, key=CLAVE + "kde_num")
        col_cat = st.selectbox("Comparar por", [None] + cols_cat, format_func=_etiqueta, key=CLAVE + "kde_cat")
        top_n = st.slider("Nº de categorías", 2, 10, 5, key=CLAVE + "kde_n", disabled=not col_cat)
        log = st.toggle("Escala logarítmica", value=True, key=CLAVE + "kde_log")
        st.caption(f"La curva se estima con una muestra de hasta {estilo.numero(gr.MUESTRA_MAX_KDE)} registros por grupo.")
    with c_graf:
        _pintar(_fig_densidad(df, col_num, col_cat, top_n, log))


@st.fragment
def _barras_frecuencia(df, cols_cat) -> None:
    c_ctrl, c_graf = st.columns([1, 3], gap="large")
    with c_ctrl:
        col = st.selectbox("Variable categórica", cols_cat, format_func=_etiqueta, key=CLAVE + "frec_col")
        top_n = st.slider("Nº de categorías", 3, 40, 15, key=CLAVE + "frec_n")
    with c_graf:
        fig = gr.barras_conteo(df, col, top_n=top_n, titulo=f"Registros por {_etiqueta(col).lower()} (top {top_n})")
        fig.update_layout(yaxis_title=_etiqueta(col))
        if col in COLUMNAS_CON_BANDERA:
            banderas_en_eje_y(fig)
        _pintar(fig, 180 + 26 * top_n)


@st.fragment
def _barras_agregacion(df, cols_num, cols_cat) -> None:
    c_ctrl, c_graf = st.columns([1, 3], gap="large")
    with c_ctrl:
        col_cat = st.selectbox("Variable categórica", cols_cat, format_func=_etiqueta, key=CLAVE + "agr_cat")
        col_num = st.selectbox("Variable numérica", cols_num, index=_indice_por_defecto(cols_num), format_func=_etiqueta, key=CLAVE + "agr_num")
        agregacion = st.segmented_control("Agregación", gr.AGREGACIONES_BARRAS, default="Media",
                                          key=CLAVE + "agr_func") or "Media"
        top_n = st.slider("Nº de categorías", 3, 40, 15, key=CLAVE + "agr_n")
        minimo = st.number_input("Mín. de registros por categoría", 1, 10_000, 30, step=10, key=CLAVE + "agr_min",
                                 help="Evita que categorías con muy pocos registros dominen el ranking.")
    with c_graf:
        conteo = df[col_cat].value_counts()
        datos = df[df[col_cat].isin(conteo[conteo >= minimo].index)]
        fig = gr.barras_agregado(datos, col_cat, col_num, agregacion=agregacion, top_n=top_n,
                                 titulo=f"{agregacion} de {_etiqueta(col_num).lower()} por {_etiqueta(col_cat).lower()}")
        fig.update_layout(yaxis_title=_etiqueta(col_cat), xaxis_title=f"{agregacion} de {_etiqueta(col_num).lower()}")
        if col_cat in COLUMNAS_CON_BANDERA:
            banderas_en_eje_y(fig)
        _pintar(fig, 180 + 26 * top_n)


@st.fragment
def _serie(df, cols_num, cols_cat, cols_fecha) -> None:
    c_ctrl, c_graf = st.columns([1, 3], gap="large")
    with c_ctrl:
        col_fecha = st.selectbox("Variable de fecha", cols_fecha, format_func=_etiqueta, key=CLAVE + "ser_fecha")
        col_num = st.selectbox("Variable numérica", cols_num, index=_indice_por_defecto(cols_num), format_func=_etiqueta, key=CLAVE + "ser_num")
        frecuencia = st.segmented_control("Periodo", list(gr._FRECUENCIAS)[2:], default="Año",
                                          key=CLAVE + "ser_freq") or "Año"
        agregacion = st.selectbox("Agregación", gr.AGREGACIONES_SERIE, key=CLAVE + "ser_agr")
        col_cat = st.selectbox("Desglosar por", [None] + cols_cat, format_func=_etiqueta, key=CLAVE + "ser_cat")
        top_n = st.slider("Nº de categorías", 2, 10, 5, key=CLAVE + "ser_n", disabled=not col_cat)
    with c_graf:
        fig = _fig_serie(df, col_fecha, col_num, col_cat, top_n, frecuencia, agregacion)
        fig.update_layout(title=f"{agregacion} de {_etiqueta(col_num).lower()} por {frecuencia.lower()}",
                          yaxis_title=f"{agregacion} de {_etiqueta(col_num).lower()}", hovermode="x unified")
        _pintar(fig)


# Opciones del filtro de valor medio del club (M€): escalonadas porque los
# valores medios van de unos pocos miles de euros a decenas de millones.
UMBRALES_VALOR_CLUB = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20]


@st.cache_data(show_spinner="Construyendo la tabla de traspasos...")
def _traspasos(_df, n_filas: int) -> pd.DataFrame:
    return rt.construir_traspasos(_df)


@st.cache_data
def _valor_medio_club(_df, n_filas: int) -> pd.Series:
    """Valor de mercado medio (€) de cada club: media de todos sus registros."""
    return _df.groupby("equipo", observed=True)["valor_mercado_eur"].mean()


@st.cache_data(show_spinner="Construyendo la red de traspasos...")
def _datos_red(_df, n_filas: int, umbral_m: float):
    """Datos de la red usando solo los traspasos en los que el club de origen
    Y el de destino tienen un valor medio de al menos `umbral_m` M€."""
    traspasos = _traspasos(_df, n_filas)
    if umbral_m > 0:
        valor = _valor_medio_club(_df, n_filas)
        validos = set(valor[valor >= umbral_m * 1e6].index)
        traspasos = traspasos[traspasos["equipo_origen"].isin(validos) & traspasos["equipo_destino"].isin(validos)]
    return rt.preparar_datos_red(traspasos.reset_index(drop=True))


@st.fragment
def _red(df) -> None:
    # Controles en una fila ENCIMA de la red: así la red ocupa todo el ancho
    # y los nodos y etiquetas tienen espacio. Todas las barras (filtros
    # numéricos) van juntas en la columna de la derecha.
    c_nivel, c_seleccion, c_ajustes = st.columns([1.1, 2.4, 1.6], gap="large")
    top_n_conexiones, top_n_equipos = None, 40

    # El valor medio mínimo va primero en la columna derecha: de él depende
    # qué ligas y equipos se ofrecen en los desplegables.
    valor_club = _valor_medio_club(df, len(df))
    with c_ajustes:
        umbral_m = st.select_slider(
            "Valor medio mínimo del club", options=UMBRALES_VALOR_CLUB, value=0,
            format_func=lambda v: "Sin filtro" if v == 0 else f"{v:g} M€".replace(".", ","),
            key=CLAVE + "red_umbral_club",
            help="Solo se usan los traspasos entre clubes cuyo valor de mercado medio (media de todos sus "
                 "registros) alcanza este valor: así se dejan fuera filiales y equipos de categorías inferiores.",
        )
        if umbral_m:
            n_validos = int((valor_club >= umbral_m * 1e6).sum())
            st.caption(f"{estilo.numero(n_validos)} de {estilo.numero(len(valor_club))} clubes lo superan.")
    aristas_equipos, liga_por_equipo, aristas_ligas, traspasos = _datos_red(df, len(df), float(umbral_m))
    lista_equipos = rt.equipos_disponibles(aristas_equipos)
    lista_ligas = rt.ligas_disponibles(aristas_equipos)
    if traspasos.empty:
        st.info("Ningún traspaso cumple el valor medio mínimo elegido. Baja el umbral.")
        return

    with c_nivel:
        nivel = st.segmented_control("Cada nodo es un...", ["Liga", "Equipo"], default="Liga",
                                     key=CLAVE + "red_nivel") or "Liga"
        modo = None
        if nivel == "Equipo":
            modo = st.radio("Elegir equipos por", ["Ligas", "Equipos"], horizontal=True, key=CLAVE + "red_modo")
        st.caption(f"{estilo.numero(len(traspasos))} traspasos entre {estilo.numero(len(lista_equipos))} equipos.")

    if nivel == "Liga":
        with c_seleccion:
            por_defecto = [p for p in ("Spain", "England", "Italy", "France", "Germany", "Portugal") if p in lista_ligas]
            ligas_sel = st.multiselect("Ligas", lista_ligas, default=por_defecto, key=CLAVE + "red_ligas",
                                       help="Con una única liga se muestran las ligas más conectadas con ella.")
        with c_ajustes:
            if len(ligas_sel) == 1:
                top_n_conexiones = st.slider("Ligas más conectadas a mostrar", 3, 30, 12, key=CLAVE + "red_top_l")
            peso = st.slider("Mín. de jugadores por enlace", 1, 50, 1, key=CLAVE + "red_peso_l")
        titulo = (f"Ligas con más traspasos con {ligas_sel[0]}" if len(ligas_sel) == 1
                  else "Red de traspasos entre ligas")
        fig, aristas = rt.red_por_ligas(aristas_ligas, ligas_sel, top_n_conexiones=top_n_conexiones,
                                        peso_minimo=peso, titulo=titulo)
        nivel_registros, col_o, col_d = "liga", "liga_origen", "liga_destino"
    else:
        equipos_sel, ligas_sel = [], []
        with c_seleccion:
            if modo == "Ligas":
                ligas_sel = st.multiselect("Ligas cuyos equipos se muestran", lista_ligas,
                                           default=[p for p in ("Spain",) if p in lista_ligas], key=CLAVE + "red_ligas_eq")
            else:
                por_defecto = [e for e in ("Real Madrid",) if e in lista_equipos]
                equipos_sel = st.multiselect("Equipos", lista_equipos, default=por_defecto, key=CLAVE + "red_equipos",
                                             help="Con un único equipo se muestran los equipos más conectados con él.")
        with c_ajustes:
            if modo == "Ligas":
                top_n_equipos = st.slider("Máx. de equipos (los más conectados)", 5, 40, 20, key=CLAVE + "red_top_e")
            elif len(equipos_sel) == 1:
                top_n_conexiones = st.slider("Equipos más conectados a mostrar", 3, 30, 12, key=CLAVE + "red_top_c")
            elif len(equipos_sel) > 1:
                top_n_equipos = None
            # Con muchos equipos a la vez, un mínimo de 3 jugadores por enlace
            # evita cientos de flechas de un único traspaso.
            peso = st.slider("Mín. de jugadores por enlace", 1, 20, 3 if modo == "Ligas" else 1,
                             key=CLAVE + ("red_peso_e_ligas" if modo == "Ligas" else "red_peso_e_equipos"))
        fig, aristas = rt.red_por_equipos(aristas_equipos, liga_por_equipo, equipos=equipos_sel or None,
                                          ligas=ligas_sel or None, top_n_conexiones=top_n_conexiones,
                                          top_n_equipos=top_n_equipos, peso_minimo=peso)
        nivel_registros, col_o, col_d = "equipo", "equipo_origen", "equipo_destino"

    st.caption("💡 Cada flecha va del origen al destino y tiene el color del origen; cuanto más gruesa, más "
               "jugadores. El tamaño del nodo indica su nº total de traspasos. Pasa el ratón por una flecha o un "
               "nodo para ver las cifras.")
    _pintar(fig, rt.ALTURA_RED)

    registros = rt.registros_de_aristas(traspasos, aristas, nivel_registros)
    resumen = rt.resumen_registros(registros)
    if resumen:
        estilo.tarjetas_kpi([
            ("Traspasos representados", estilo.numero(resumen["n_traspasos"]),
             f"{resumen['fecha_min']:%Y} – {resumen['fecha_max']:%Y}"),
            ("Jugadores distintos", estilo.numero(resumen["n_jugadores"]), ""),
            ("Valor medio al traspasar", estilo.euros(resumen["valor_medio"]),
             f"Total: {estilo.euros(resumen['valor_total'])}"),
            ("Fichaje más valioso", estilo.euros(resumen["valor_mas_valioso"]), str(resumen["jugador_mas_valioso"])),
        ])
        st.write("")

    tab_desglose, tab_jugadores = st.tabs(["↔️ Desglose por dirección", "👥 Jugadores implicados"])
    with tab_desglose:
        etiqueta = "Liga" if nivel_registros == "liga" else "Equipo"
        desglose = rt.tabla_desglose_direcciones(aristas, col_o, col_d).rename(
            columns={"Origen": f"{etiqueta} origen", "Destino": f"{etiqueta} destino"})
        if desglose.empty:
            st.info("No hay combinaciones con traspasos que mostrar.")
        else:
            st.dataframe(desglose, hide_index=True, use_container_width=True,
                         column_config={"Nº de fichajes": st.column_config.ProgressColumn(
                             "Nº de fichajes", format="%d", min_value=0, max_value=int(desglose["Nº de fichajes"].max()))})
    with tab_jugadores:
        busqueda = st.text_input("Buscar jugador", key=CLAVE + "red_busqueda", placeholder="Nombre")
        tabla = registros
        if busqueda:
            tabla = tabla[tabla["nombre"].str.contains(busqueda, case=False, regex=False, na=False)]
        st.dataframe(
            tabla.sort_values("valor_mercado", ascending=False)[
                ["nombre", "equipo_origen", "equipo_destino", "liga_origen", "liga_destino", "fecha", "valor_mercado"]],
            hide_index=True, use_container_width=True, height=380,
            column_config={
                "nombre": "Jugador", "equipo_origen": "Equipo origen", "equipo_destino": "Equipo destino",
                "liga_origen": "Liga origen", "liga_destino": "Liga destino",
                "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                "valor_mercado": st.column_config.NumberColumn("Valor de mercado", format="euro"),
            },
        )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
def render() -> None:
    df = cargar_tabla()
    estilo.cabecera(
        "Visualización de datos",
        "Explora la forma de las variables y los flujos de traspasos entre ligas y equipos.",
    )

    cols_num = gr.columnas_numericas(df)
    cols_cat = gr.columnas_categoricas(df)
    cols_fecha = gr.columnas_fecha(df)

    tipo = st.segmented_control(
        "Tipo de gráfica", list(TIPOS), default="Boxplot", format_func=lambda t: f"{TIPOS[t][0]} {t}",
        key=CLAVE + "tipo", label_visibility="collapsed",
    ) or "Boxplot"
    st.caption(TIPOS[tipo][1])
    st.divider()

    if tipo == "Boxplot":
        _boxplot(df, cols_num, cols_cat)
    elif tipo == "Densidad (KDE)":
        _densidad(df, cols_num, cols_cat)
    elif tipo == "Barras — frecuencia":
        _barras_frecuencia(df, cols_cat)
    elif tipo == "Barras — agregación":
        _barras_agregacion(df, cols_num, cols_cat)
    elif tipo == "Serie temporal":
        _serie(df, cols_num, cols_cat, cols_fecha)
    else:
        _red(df)
