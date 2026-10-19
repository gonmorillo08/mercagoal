"""
Red dirigida de traspasos de jugadores.

Este módulo es independiente de `graficas.py` y se encarga en exclusiva de
DIBUJAR la red de traspasos entre equipos y/o entre ligas.

Copia del módulo homónimo del proyecto de administración (TFG), adaptada a
la tabla LIMPIA: allí la red se leía de `traspasos.csv` (construida a partir
de la tabla original); aquí se construye directamente a partir de la tabla
limpia (`construir_traspasos()`), con el mismo criterio: un traspaso cada
vez que el equipo de un jugador cambia entre dos valoraciones consecutivas.
Así los equipos de la red son exactamente los de la tabla limpia (ya
fusionados duplicados y canteras) y la "liga" se identifica por el país de la competición.

La red admite DOS niveles de nodo, elegibles desde la interfaz:

  - Nodo = LIGA: cada nodo es una liga; un enlace liga_A -> liga_B indica
    que uno o más jugadores han pasado directamente de un equipo de la
    liga A a un equipo de la liga B. Solo se muestran (como nodos) las
    ligas que el usuario seleccione.
        -> resumen_transferencias_ligas() + red_por_ligas()

  - Nodo = EQUIPO: cada nodo es un equipo; un enlace equipo_A -> equipo_B
    indica que uno o más jugadores han pasado directamente de A a B. Los
    equipos a mostrar se pueden elegir de dos formas: indicando
    directamente los equipos, o indicando las ligas cuyos equipos se
    quieren mostrar.
        -> resumen_transferencias_equipos() + red_por_equipos()

En ambos casos, el sentido del traspaso se dibuja con una flecha y el
tamaño de cada nodo depende de su nº de enlaces (importancia).
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from comun.banderas import url_bandera

# Altura (px) con la que se pinta la red en la app; se usa para convertir el
# tamaño de los nodos (px) a unidades del gráfico al colocar las banderas.
ALTURA_RED = 720


# ---------------------------------------------------------------------------
# Construcción de la tabla de traspasos a partir de la tabla limpia
# ---------------------------------------------------------------------------


def construir_traspasos(df: pd.DataFrame) -> pd.DataFrame:
    """Un traspaso por cada cambio de equipo entre dos valoraciones
    consecutivas del mismo jugador (mismo criterio que
    `construir_historial_transferencias()` del proyecto TFG). Columnas:
    `id_player, nombre, equipo_origen, equipo_destino, liga_origen,
    liga_destino, fecha, valor_mercado` (valor de mercado del jugador en el
    registro en el que se detecta el cambio)."""
    datos = df[["id_player", "nombre", "equipo", "liga", "fecha", "valor_mercado_eur"]].dropna(
        subset=["id_player", "equipo", "fecha"]
    )
    datos = datos.sort_values(["id_player", "fecha"], kind="mergesort")
    equipo_anterior = datos.groupby("id_player", sort=False)["equipo"].shift(1)
    liga_anterior = datos.groupby("id_player", sort=False)["liga"].shift(1)
    hay_cambio = equipo_anterior.notna() & (datos["equipo"] != equipo_anterior)
    return pd.DataFrame(
        {
            "id_player": datos.loc[hay_cambio, "id_player"].values,
            "nombre": datos.loc[hay_cambio, "nombre"].values,
            "equipo_origen": equipo_anterior[hay_cambio].values,
            "equipo_destino": datos.loc[hay_cambio, "equipo"].values,
            "liga_origen": liga_anterior[hay_cambio].values,
            "liga_destino": datos.loc[hay_cambio, "liga"].values,
            "fecha": pd.to_datetime(datos.loc[hay_cambio, "fecha"].values),
            "valor_mercado": datos.loc[hay_cambio, "valor_mercado_eur"].values,
        }
    ).reset_index(drop=True)


def _liga_por_equipo(traspasos: pd.DataFrame) -> dict:
    """Liga más habitual de cada equipo (para colorear/filtrar por liga en
    la red de equipos), calculada a partir de la propia tabla de
    traspasos: para cada traspaso, el equipo de origen "vota" por su
    `liga_origen` y el de destino por su `liga_destino`."""
    filas = pd.concat(
        [
            traspasos[["equipo_origen", "liga_origen"]].rename(
                columns={"equipo_origen": "equipo", "liga_origen": "liga"}
            ),
            traspasos[["equipo_destino", "liga_destino"]].rename(
                columns={"equipo_destino": "equipo", "liga_destino": "liga"}
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["liga"])

    return (
        filas.groupby("equipo")["liga"]
        .agg(lambda serie: serie.mode().iat[0] if not serie.mode().empty else None)
        .to_dict()
    )


def resumen_transferencias_equipos(
    traspasos: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Agrega los traspasos individuales (de `construir_traspasos()`) en
    enlaces equipo_origen -> equipo_destino con su peso (nº de jugadores
    que han hecho ese traspaso directo), y calcula la liga más habitual de
    cada equipo (para colorear/filtrar por liga en la red de equipos)."""
    if traspasos is None:
        raise ValueError("Pasa la tabla de traspasos (ver construir_traspasos(df)).")

    liga_por_equipo = _liga_por_equipo(traspasos)

    columnas = ["equipo_origen", "equipo_destino", "peso", "liga_origen", "liga_destino"]
    if traspasos.empty:
        return pd.DataFrame(columns=columnas), liga_por_equipo

    aristas = (
        traspasos.groupby(["equipo_origen", "equipo_destino"])
        .agg(peso=("id_player", "count"))
        .reset_index()
    )
    aristas["liga_origen"] = aristas["equipo_origen"].map(liga_por_equipo)
    aristas["liga_destino"] = aristas["equipo_destino"].map(liga_por_equipo)
    aristas = aristas.sort_values("peso", ascending=False).reset_index(drop=True)
    return aristas[columnas], liga_por_equipo


def resumen_transferencias_ligas(traspasos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Agrega los traspasos individuales (de `construir_traspasos()`) en
    enlaces liga_origen -> liga_destino con su peso (nº de jugadores que
    han cambiado de una liga a otra directamente). Solo se cuentan los
    traspasos en los que se conoce la liga de origen y de destino."""
    if traspasos is None:
        raise ValueError("Pasa la tabla de traspasos (ver construir_traspasos(df)).")

    columnas = ["liga_origen", "liga_destino", "peso"]
    traspasos_con_liga = traspasos.dropna(subset=["liga_origen", "liga_destino"])
    if traspasos_con_liga.empty:
        return pd.DataFrame(columns=columnas)

    aristas = (
        traspasos_con_liga.groupby(["liga_origen", "liga_destino"])
        .agg(peso=("id_player", "count"))
        .reset_index()
        .sort_values("peso", ascending=False)
        .reset_index(drop=True)
    )
    return aristas[columnas]


def preparar_datos_red(
    traspasos: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Calcula de una vez todo lo necesario para la red (equipos y ligas)
    a partir de la tabla de traspasos ya generada (`construir_traspasos()`).
    Pensada para cachearse en la app (p. ej. con `st.cache_data`), ya que
    no depende de los filtros interactivos que elija el usuario (nivel de
    nodo, ligas/equipos concretos...): esos se aplican después, en
    `red_por_ligas()` / `red_por_equipos()`.

    Devuelve `(aristas_equipos, liga_por_equipo, aristas_ligas, traspasos)`.
    `traspasos` es el detalle individual (un jugador por fila) que hay
    detrás de cada enlace agregado; se usa para mostrar la tabla y el
    resumen de jugadores implicados en la red que se esté visualizando.
    """
    if traspasos is None:
        raise ValueError("Pasa la tabla de traspasos (ver construir_traspasos(df)).")

    liga_por_equipo = _liga_por_equipo(traspasos)

    columnas_equipos = ["equipo_origen", "equipo_destino", "peso", "liga_origen", "liga_destino"]
    columnas_ligas = ["liga_origen", "liga_destino", "peso"]

    if traspasos.empty:
        return (
            pd.DataFrame(columns=columnas_equipos),
            liga_por_equipo,
            pd.DataFrame(columns=columnas_ligas),
            traspasos,
        )

    aristas_equipos = (
        traspasos.groupby(["equipo_origen", "equipo_destino"])
        .agg(peso=("id_player", "count"))
        .reset_index()
    )
    aristas_equipos["liga_origen"] = aristas_equipos["equipo_origen"].map(liga_por_equipo)
    aristas_equipos["liga_destino"] = aristas_equipos["equipo_destino"].map(liga_por_equipo)
    aristas_equipos = aristas_equipos.sort_values("peso", ascending=False).reset_index(drop=True)

    traspasos_con_liga = traspasos.dropna(subset=["liga_origen", "liga_destino"])
    if traspasos_con_liga.empty:
        aristas_ligas = pd.DataFrame(columns=columnas_ligas)
    else:
        aristas_ligas = (
            traspasos_con_liga.groupby(["liga_origen", "liga_destino"])
            .agg(peso=("id_player", "count"))
            .reset_index()
            .sort_values("peso", ascending=False)
            .reset_index(drop=True)
        )

    return (
        aristas_equipos[columnas_equipos],
        liga_por_equipo,
        aristas_ligas[columnas_ligas],
        traspasos,
    )


# ---------------------------------------------------------------------------
# Utilidades para poblar los selectores de la interfaz
# ---------------------------------------------------------------------------


def equipos_disponibles(aristas_equipos: pd.DataFrame) -> list[str]:
    """Lista ordenada de todos los equipos presentes en al menos un
    traspaso (origen o destino); útil para poblar un `multiselect`."""
    return sorted(set(aristas_equipos["equipo_origen"]) | set(aristas_equipos["equipo_destino"]))


def ligas_disponibles(aristas_equipos: pd.DataFrame) -> list[str]:
    """Lista ordenada de las ligas presentes en los traspasos (origen o
    destino de algún equipo), sin incluir nulos; útil para poblar un
    `multiselect`, tanto para la red de ligas como para filtrar equipos
    por liga en la red de equipos."""
    ligas = set(aristas_equipos["liga_origen"].dropna()) | set(aristas_equipos["liga_destino"].dropna())
    return sorted(ligas)


# ---------------------------------------------------------------------------
# Registros de jugadores detrás de una red ya construida (tabla + resumen)
# ---------------------------------------------------------------------------


def registros_de_aristas(
    traspasos: pd.DataFrame,
    aristas_mostradas: pd.DataFrame,
    nivel: str,
) -> pd.DataFrame:
    """Dado el detalle individual de traspasos (de `preparar_datos_red`) y
    las aristas que se están dibujando en un gráfico concreto (de
    `red_por_ligas` / `red_por_equipos`), devuelve las filas de jugadores
    que sustentan exactamente esos enlaces — para poder listarlas/buscarlas
    debajo del gráfico.

    `nivel` indica si las aristas son de nivel "liga" o "equipo".
    """
    columnas = list(traspasos.columns)
    if aristas_mostradas.empty:
        return traspasos.iloc[0:0]

    if nivel == "equipo":
        pares = aristas_mostradas[["equipo_origen", "equipo_destino"]].drop_duplicates()
        registros = traspasos.merge(pares, on=["equipo_origen", "equipo_destino"], how="inner")
    else:
        pares = aristas_mostradas[["liga_origen", "liga_destino"]].drop_duplicates()
        registros = traspasos.dropna(subset=["liga_origen", "liga_destino"]).merge(
            pares, on=["liga_origen", "liga_destino"], how="inner"
        )
    return registros[columnas]


def resumen_registros(
    registros: pd.DataFrame,
    columna_valor: str = "valor_mercado",
    columna_nombre: str = "nombre",
) -> dict:
    """Pequeño resumen (jugador más valioso, nº de traspasos y jugadores
    distintos, valor medio, periodo cubierto...) de un conjunto de
    registros de traspasos, para mostrar junto a la tabla, basado en
    `valor_mercado` (valor de mercado del jugador en la fecha del
    traspaso)."""
    registros = registros.dropna(subset=[columna_valor])
    if registros.empty:
        return {}

    idx_max = registros[columna_valor].idxmax()

    return {
        "n_traspasos": int(len(registros)),
        "n_jugadores": int(registros["id_player"].nunique()),
        "jugador_mas_valioso": registros.loc[idx_max, columna_nombre],
        "valor_mas_valioso": float(registros.loc[idx_max, columna_valor]),
        "valor_medio": float(registros[columna_valor].mean()),
        "valor_total": float(registros[columna_valor].sum()),
        "fecha_min": registros["fecha"].min(),
        "fecha_max": registros["fecha"].max(),
    }


# ---------------------------------------------------------------------------
# Utilidades internas de dibujado (compartidas por ambos niveles de red)
# ---------------------------------------------------------------------------


def _paleta_por_categoria(categorias: list[str]) -> dict[str, str]:
    paleta = px.colors.qualitative.Set2 + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    color_por_categoria = {cat: paleta[i % len(paleta)] for i, cat in enumerate(categorias)}
    return color_por_categoria


def _funcion_tamano(grado: pd.Series):
    grado_min, grado_max = grado.min(), grado.max()

    def _tamano(valor_grado: float) -> float:
        if grado_max == grado_min:
            return 27.5
        elif grado_max - grado_min < 10:
            return 30 + (valor_grado - grado_min) / (grado_max - grado_min) * 20.0
        
        # Mínimo de 20 px (antes 14) para que ningún nodo se pierda.
        return 20 + (valor_grado - grado_min) / (grado_max - grado_min) * 38.0

    return _tamano


def _funcion_ancho(pesos: pd.Series):
    peso_min, peso_max = pesos.min(), pesos.max()

    def _ancho(peso: float) -> float:
        if peso_max == peso_min:
            return 2.0
        return 1.0 + (peso - peso_min) / (peso_max - peso_min) * 5.0

    return _ancho


def _grado_por_nodo(aristas: pd.DataFrame, col_origen: str, col_destino: str) -> pd.Series:
    """Nº de enlaces (entrantes + salientes, ponderados por `peso`) de
    cada nodo: es la medida de "importancia" usada para el tamaño."""
    return (
        pd.concat(
            [aristas.groupby(col_origen)["peso"].sum(), aristas.groupby(col_destino)["peso"].sum()],
            axis=1,
        )
        .fillna(0)
        .sum(axis=1)
    )


def _aristas_conectadas_a(
    aristas: pd.DataFrame,
    col_origen: str,
    col_destino: str,
    nodo_central: str,
    top_n: int | None,
    peso_minimo: int,
) -> pd.DataFrame:
    """Enlaces (en ambos sentidos) de `nodo_central` con el resto de
    nodos. Si se indica `top_n`, se queda solo con los `top_n` nodos con
    más enlaces (sumando entrantes + salientes) respecto a `nodo_central`
    — usado para expandir un único nodo seleccionado a su "ego network"."""
    aristas_f = aristas[aristas["peso"] >= peso_minimo]
    tocan_central = aristas_f[
        (aristas_f[col_origen] == nodo_central) | (aristas_f[col_destino] == nodo_central)
    ]
    # Se excluyen los auto-enlaces (p. ej. traspasos dentro de la misma
    # liga): no representan una conexión con "otro" nodo.
    tocan_central = tocan_central[tocan_central[col_origen] != tocan_central[col_destino]]

    if tocan_central.empty or not top_n:
        return tocan_central

    vecino = np.where(
        tocan_central[col_origen] == nodo_central,
        tocan_central[col_destino],
        tocan_central[col_origen],
    )
    tocan_central = tocan_central.assign(_vecino=vecino)
    vecinos_top = set(tocan_central.groupby("_vecino")["peso"].sum().nlargest(top_n).index)
    return tocan_central[tocan_central["_vecino"].isin(vecinos_top)].drop(columns="_vecino")


def _subgrafo_estricto(
    aristas: pd.DataFrame,
    col_origen: str,
    col_destino: str,
    nodos: set[str] | list[str],
    peso_minimo: int,
    excluir_autoenlaces: bool = False,
) -> pd.DataFrame:
    """Todos los enlaces (en ambos sentidos, es decir TODAS las
    combinaciones dirigidas) entre los nodos indicados: la base tanto del
    gráfico como del desglose de traspasos por dirección."""
    nodos_set = set(nodos)
    aristas_f = aristas[aristas["peso"] >= peso_minimo]
    aristas_f = aristas_f[aristas_f[col_origen].isin(nodos_set) & aristas_f[col_destino].isin(nodos_set)]
    if excluir_autoenlaces:
        aristas_f = aristas_f[aristas_f[col_origen] != aristas_f[col_destino]]
    return aristas_f


def tabla_desglose_direcciones(
    aristas_mostradas: pd.DataFrame,
    col_origen: str,
    col_destino: str,
) -> pd.DataFrame:
    """Desglose del nº de fichajes en cada dirección entre los nodos que
    aparecen en un gráfico ya construido (p. ej. si se han elegido
    Premier League y LaLiga: cuántos fichajes LaLiga -> Premier League y
    cuántos Premier League -> LaLiga). Es, literalmente, la tabla de
    `aristas_mostradas` ya que ese DataFrame contiene un enlace dirigido
    por cada combinación con al menos un traspaso."""
    columnas = [col_origen, col_destino, "peso"]
    if aristas_mostradas.empty:
        return pd.DataFrame(columns=["Origen", "Destino", "Nº de fichajes"])
    tabla = (
        aristas_mostradas[columnas]
        .sort_values("peso", ascending=False)
        .reset_index(drop=True)
        .rename(columns={col_origen: "Origen", col_destino: "Destino", "peso": "Nº de fichajes"})
    )
    return tabla


def _figura_vacia(mensaje: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=mensaje)
    return fig


def _dibujar_red(
    aristas: pd.DataFrame,
    col_origen: str,
    col_destino: str,
    nodos: list[str],
    grado: pd.Series,
    color_por_nodo: dict[str, str],
    grupo_por_nodo: dict[str, str] | None,
    titulo: str,
    semilla: int,
    nodos_destacados: set[str] | None = None,
    banderas: bool = False,
) -> go.Figure:
    """Mismo diseño que la red del proyecto de administración (disposición
    libre de networkx, colores suaves por liga, flechas del color del
    origen, tamaño según nº de enlaces), con ajustes de legibilidad:
      - Nodos más separados (más distancia en el 'spring layout').
      - Etiquetas FUERA del nodo, en texto oscuro (antes, blanco de 9 pt
        dentro de nodos pastel), y un borde gris fino en cada nodo.
      - Flechas que empiezan y terminan en el borde de cada nodo (en vez de
        a un 90 % fijo del recorrido), y ligeramente separadas cuando hay
        traspasos en los dos sentidos.
      - Márgenes calculados para que ninguna etiqueta quede cortada.
    """
    grafo = nx.DiGraph()
    grafo.add_nodes_from(nodos)
    for _, fila in aristas.iterrows():
        grafo.add_edge(fila[col_origen], fila[col_destino], peso=fila["peso"])

    # Hasta 60 nodos, Kamada-Kawai: disposición libre como el 'spring
    # layout' original, pero reparte los nodos de forma más uniforme y no
    # amontona el resto cuando algún nodo tiene muy pocos enlaces (p. ej.
    # al filtrar por valor medio del club). Con más nodos, 'spring layout'.
    pos = None
    if 2 < len(nodos) <= 60:
        try:
            pos = nx.kamada_kawai_layout(grafo.to_undirected())
        except Exception:  # p. ej. grafos degenerados
            pos = None
    if pos is None:
        k = 2.4 / max(len(nodos), 1) ** 0.5
        pos = nx.spring_layout(grafo, weight="peso", k=k, seed=semilla, iterations=300)

    tamano_nodo = _funcion_tamano(grado)
    ancho_enlace = _funcion_ancho(aristas["peso"]) if not aristas.empty else (lambda p: 2.0)

    fig = go.Figure()

    # -- Enlaces dirigidos: una flecha (annotation) por traspaso agregado --
    anotaciones = []
    hover_x, hover_y, hover_texto = [], [], []
    for _, fila in aristas.sort_values("peso").iterrows():
        origen, destino, peso = fila[col_origen], fila[col_destino], fila["peso"]
        if origen == destino:
            continue
        (x0, y0), (x1, y1) = pos[origen], pos[destino]
        # Desplazamiento perpendicular pequeño: A->B y B->A quedan paralelas.
        dx, dy = x1 - x0, y1 - y0
        largo = max((dx ** 2 + dy ** 2) ** 0.5, 1e-9)
        ox, oy = dy / largo * 0.012, -dx / largo * 0.012
        anotaciones.append(
            dict(
                x=x1 + ox, y=y1 + oy, ax=x0 + ox, ay=y0 + oy,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=ancho_enlace(peso),
                arrowcolor=color_por_nodo[origen], opacity=0.6,
                # En píxeles: la flecha empieza/acaba en el borde de cada nodo.
                standoff=tamano_nodo(grado[destino]) / 2 + 1,
                startstandoff=tamano_nodo(grado[origen]) / 2 + 1,
            )
        )
        hover_x.append((x0 + x1) / 2)
        hover_y.append((y0 + y1) / 2)
        hover_texto.append(f"<b>{origen} → {destino}</b><br>{int(peso)} jugador(es) traspasado(s)")

    # Traza invisible sobre el punto medio de cada enlace, solo para poder
    # ver el nº de traspasos al pasar el ratón.
    fig.add_trace(go.Scatter(
        x=hover_x, y=hover_y, mode="markers", marker=dict(size=10, color="rgba(0,0,0,0)"),
        hoverinfo="text", hovertext=hover_texto, showlegend=False,
    ))

    # -- Nodos --
    # Etiquetas de todos los nodos si la red es pequeña; si es grande, solo
    # de los 30 más conectados (y de los elegidos por el usuario).
    con_etiqueta = set(nodos) if len(nodos) <= 40 else set(grado.nlargest(30).index) | (nodos_destacados or set())
    if grupo_por_nodo:
        for grupo in sorted(set(grupo_por_nodo.values())):
            nodos_grupo = [n for n in nodos if grupo_por_nodo[n] == grupo]
            if nodos_grupo:
                _añadir_trace_nodos(fig, nodos_grupo, pos, grado, tamano_nodo, color_por_nodo, con_etiqueta,
                                    nombre=grupo, grupo_label=grupo, nodos_destacados=nodos_destacados)
    else:
        _añadir_trace_nodos(fig, nodos, pos, grado, tamano_nodo, color_por_nodo, con_etiqueta,
                            nombre=None, grupo_label=None, nodos_destacados=nodos_destacados)

    # Márgenes: un 18 % extra en horizontal y un 12 % en vertical sobre el
    # recorrido de los nodos, para que quepan las etiquetas.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ancho_x = (max(xs) - min(xs)) or 1.0
    ancho_y = (max(ys) - min(ys)) or 1.0
    fig.update_layout(
        title=titulo,
        annotations=anotaciones,
        showlegend=bool(grupo_por_nodo),
        legend_title_text="Liga" if grupo_por_nodo else None,
        xaxis=dict(visible=False, range=[min(xs) - 0.18 * ancho_x, max(xs) + 0.18 * ancho_x]),
        yaxis=dict(visible=False, range=[min(ys) - 0.10 * ancho_y, max(ys) + 0.14 * ancho_y]),
        hovermode="closest",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=60, b=10),
        height=ALTURA_RED,
    )
    if banderas:
        _añadir_banderas_nodos(fig, nodos, pos, grado, tamano_nodo, alto_y=ancho_y * 1.24)
    return fig


def _añadir_banderas_nodos(fig: go.Figure, nodos: list[str], pos: dict, grado: pd.Series, tamano_nodo,
                           alto_y: float) -> None:
    """Bandera dentro de cada nodo (cuando los nodos son ligas/federaciones),
    ocupando ~60 % de su diámetro. El diámetro está en píxeles: se convierte a
    unidades del eje Y con la altura útil del gráfico (altura - márgenes)."""
    unidades_por_px = alto_y / (ALTURA_RED - 70)
    for nodo in nodos:
        url = url_bandera(nodo)
        if not url:
            continue
        alto = tamano_nodo(grado[nodo]) * 0.6 * unidades_por_px
        fig.add_layout_image(dict(
            source=url, xref="x", yref="y", x=pos[nodo][0], y=pos[nodo][1],
            sizex=alto * 4, sizey=alto, xanchor="center", yanchor="middle", sizing="contain", layer="above",
        ))


def _añadir_trace_nodos(
    fig: go.Figure,
    nodos: list[str],
    pos: dict,
    grado: pd.Series,
    tamano_nodo,
    color_por_nodo: dict[str, str],
    con_etiqueta: set[str],
    nombre: str | None,
    grupo_label: str | None,
    nodos_destacados: set[str] | None = None,
) -> None:
    nodos_destacados = nodos_destacados or set()
    xs = [pos[n][0] for n in nodos]
    ys = [pos[n][1] for n in nodos]
    textos = [f"<b>{n}</b>" if n in nodos_destacados else (n if n in con_etiqueta else "") for n in nodos]
    if grupo_label:
        hover = [
            f"<b>{n}</b><br>Liga: {grupo_label}<br>Traspasos (entrantes + salientes): {int(grado[n])}"
            for n in nodos
        ]
    else:
        hover = [f"<b>{n}</b><br>Traspasos (entrantes + salientes): {int(grado[n])}" for n in nodos]

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=textos,
            textposition="top center",
            textfont=dict(size=12, color="#1F2937"),
            marker=dict(
                size=[tamano_nodo(grado[n]) for n in nodos],
                color=[color_por_nodo[n] for n in nodos],
                line=dict(
                    width=[3 if n in nodos_destacados else 1 for n in nodos],
                    color=["#222222" if n in nodos_destacados else "#64748B" for n in nodos],
                ),
            ),
            name=nombre,
            showlegend=nombre is not None,
            hoverinfo="text",
            hovertext=hover,
            cliponaxis=False,
        )
    )


# ---------------------------------------------------------------------------
# Red con nodo = LIGA
# ---------------------------------------------------------------------------


def red_por_ligas(
    aristas_ligas: pd.DataFrame,
    ligas: list[str],
    top_n_conexiones: int | None = None,
    peso_minimo: int = 1,
    semilla: int = 42,
    titulo: str | None = None,
) -> tuple[go.Figure, pd.DataFrame]:
    """Red DIRIGIDA en la que cada nodo es una LIGA.

    Un enlace liga_A -> liga_B indica que uno o más jugadores han pasado
    directamente de un equipo de la liga A a un equipo de la liga B.

    Comportamiento según cuántas ligas se seleccionen en `ligas`:
      - Si se selecciona EXACTAMENTE UNA liga, se expande automáticamente
        a esa liga más las `top_n_conexiones` ligas con más enlaces con
        ella (sumando entrantes y salientes) — una "red ego" centrada en
        esa liga. El gráfico se centra ÚNICAMENTE en los enlaces que
        tocan a la liga elegida (no se muestran enlaces entre dos ligas
        "vecinas" entre sí).
      - Si se seleccionan DOS O MÁS ligas, se muestran EXCLUSIVAMENTE los
        nodos de las ligas seleccionadas y TODAS las combinaciones
        dirigidas de traspasos entre ellas, aunque no tengan enlaces
        entre sí, para que el usuario vea exactamente lo que ha elegido.

    En ambos casos se excluyen los traspasos dentro de la misma liga
    (liga_origen == liga_destino): no representan una conexión ENTRE dos
    ligas distintas.

    Devuelve `(figura, aristas_mostradas)`: `aristas_mostradas` es el
    DataFrame de enlaces realmente dibujados, útil tanto para el desglose
    por dirección (`tabla_desglose_direcciones()`) como para extraer los
    registros de jugadores que hay detrás (`registros_de_aristas()`).
    """
    if not ligas:
        return _figura_vacia("Selecciona al menos una liga para generar la red."), aristas_ligas.iloc[0:0]

    ligas_sel = list(dict.fromkeys(ligas))  # sin duplicados, conserva el orden

    if len(ligas_sel) == 1:
        nodo_central = ligas_sel[0]
        aristas_f = _aristas_conectadas_a(
            aristas_ligas, "liga_origen", "liga_destino", nodo_central, top_n_conexiones, peso_minimo
        )
        nodos = sorted({nodo_central} | set(aristas_f["liga_origen"]) | set(aristas_f["liga_destino"]))
        titulo_defecto = f"Ligas con más traspasos con {nodo_central}"
    else:
        nodos = sorted(set(ligas_sel))
        aristas_f = _subgrafo_estricto(
            aristas_ligas, "liga_origen", "liga_destino", nodos, peso_minimo, excluir_autoenlaces=True
        )
        titulo_defecto = "Red dirigida de traspasos entre ligas"

    grado = _grado_por_nodo(aristas_f, "liga_origen", "liga_destino").reindex(nodos).fillna(0)
    color_por_nodo = _paleta_por_categoria(nodos)

    fig = _dibujar_red(
        aristas=aristas_f,
        col_origen="liga_origen",
        col_destino="liga_destino",
        nodos=nodos,
        grado=grado,
        color_por_nodo=color_por_nodo,
        grupo_por_nodo=None,
        titulo=titulo or titulo_defecto,
        semilla=semilla,
        nodos_destacados=set(ligas_sel),
        banderas=True,
    )
    return fig, aristas_f


# ---------------------------------------------------------------------------
# Red con nodo = EQUIPO
# ---------------------------------------------------------------------------


def red_por_equipos(
    aristas_equipos: pd.DataFrame,
    liga_por_equipo: dict,
    equipos: list[str] | None = None,
    ligas: list[str] | None = None,
    top_n_conexiones: int | None = None,
    top_n_equipos: int = 40,
    peso_minimo: int = 1,
    semilla: int = 42,
    titulo: str | None = None,
) -> tuple[go.Figure, pd.DataFrame]:
    """Red DIRIGIDA en la que cada nodo es un EQUIPO.

    Un enlace equipo_A -> equipo_B indica que uno o más jugadores han
    pasado directamente de A a B. El tamaño de cada nodo depende de su nº
    de enlaces (traspasos entrantes + salientes) y su color de la liga más
    habitual del equipo.

    Los equipos a mostrar se eligen de una de estas dos formas
    (excluyentes; se indica solo una desde la app):
      - `ligas`: se muestran los equipos cuya liga habitual esté entre las
        indicadas y TODAS las combinaciones dirigidas de traspasos entre
        ellos; si hay más de `top_n_equipos`, se muestran solo los más
        conectados.
      - `equipos`: 
          - Si se indica EXACTAMENTE UN equipo, se expande
            automáticamente a ese equipo más los `top_n_conexiones`
            equipos con más enlaces con él (sumando entrantes y
            salientes) — una "red ego" centrada en ese equipo. El gráfico
            se centra ÚNICAMENTE en los enlaces que tocan al equipo
            elegido (no se muestran enlaces entre dos equipos "vecinos"
            entre sí).
          - Si se indican DOS O MÁS equipos, se muestran EXCLUSIVAMENTE
            los equipos indicados y TODAS las combinaciones dirigidas de
            traspasos entre ellos.

    Devuelve `(figura, aristas_mostradas)`: `aristas_mostradas` es el
    DataFrame de enlaces realmente dibujados, útil tanto para el desglose
    por dirección (`tabla_desglose_direcciones()`) como para extraer los
    registros de jugadores que hay detrás (`registros_de_aristas()`).
    """
    if not equipos and not ligas:
        vacio = aristas_equipos.iloc[0:0]
        return _figura_vacia("Selecciona al menos una liga o un equipo para generar la red."), vacio

    aristas_base = aristas_equipos[aristas_equipos["peso"] >= peso_minimo]
    nodos_destacados: set[str] = set()

    if ligas:
        # Todos los equipos cuya liga habitual está en la selección, y
        # todas las combinaciones de traspasos entre ellos.
        ligas_sel = set(ligas)
        aristas_f = aristas_base[
            aristas_base["liga_origen"].isin(ligas_sel) & aristas_base["liga_destino"].isin(ligas_sel)
        ]
        nodos = sorted(set(aristas_f["equipo_origen"]) | set(aristas_f["equipo_destino"]))
    elif len(equipos) == 1:
        # Un único equipo: se centra la red únicamente en los enlaces que
        # tocan a ese equipo (con los equipos que más enlaces tienen con
        # él), sin añadir enlaces entre dos de esos equipos "vecinos".
        nodo_central = equipos[0]
        aristas_f = _aristas_conectadas_a(
            aristas_base, "equipo_origen", "equipo_destino", nodo_central, top_n_conexiones, peso_minimo
        )
        nodos = sorted({nodo_central} | set(aristas_f["equipo_origen"]) | set(aristas_f["equipo_destino"]))
        nodos_destacados = {nodo_central}
    else:
        # Dos o más equipos: exclusivamente esos equipos y todas las
        # combinaciones de traspasos entre ellos.
        nodos = sorted(set(equipos))
        aristas_f = _subgrafo_estricto(aristas_equipos, "equipo_origen", "equipo_destino", nodos, peso_minimo)
        nodos_destacados = set(equipos)

    if not nodos and equipos and len(equipos) == 1:
        # Equipo central sin ninguna conexión: se muestra igualmente, aislado.
        nodos = [equipos[0]]

    if not nodos:
        vacio = aristas_equipos.iloc[0:0]
        return _figura_vacia("No hay traspasos que cumplan los filtros seleccionados."), vacio

    grado = _grado_por_nodo(aristas_f, "equipo_origen", "equipo_destino").reindex(nodos).fillna(0)

    # Límite de seguridad para no saturar el gráfico (solo relevante al
    # filtrar por liga o al escoger muchos equipos a la vez; en el modo
    # "un único equipo" el nº de conexiones ya lo controla
    # `top_n_conexiones`, así que no se vuelve a recortar aquí).
    es_modo_ego = bool(equipos) and len(equipos) == 1
    if not es_modo_ego and top_n_equipos and len(grado) > top_n_equipos:
        equipos_top = set(grado.nlargest(top_n_equipos).index)
        if equipos:
            equipos_top |= set(equipos) & set(grado.index)
        aristas_f = aristas_f[
            aristas_f["equipo_origen"].isin(equipos_top)
            & aristas_f["equipo_destino"].isin(equipos_top)
        ]
        grado = grado.loc[grado.index.isin(equipos_top)]
        nodos = sorted(equipos_top)

    if equipos and len(equipos) == 1:
        titulo_defecto = f"Equipos con más traspasos con {equipos[0]}"
    else:
        titulo_defecto = "Red dirigida de traspasos entre equipos"

    grado = grado.reindex(nodos).fillna(0)

    liga_de_nodo = {equipo: (liga_por_equipo.get(equipo) or "Liga desconocida") for equipo in nodos}
    ligas_unicas = sorted(set(liga_de_nodo.values()))
    paleta_ligas = _paleta_por_categoria(ligas_unicas)
    if "Liga desconocida" in paleta_ligas:
        paleta_ligas["Liga desconocida"] = "#B0B0B0"
    color_por_nodo = {equipo: paleta_ligas[liga_de_nodo[equipo]] for equipo in nodos}

    fig = _dibujar_red(
        aristas=aristas_f,
        col_origen="equipo_origen",
        col_destino="equipo_destino",
        nodos=nodos,
        grado=grado,
        color_por_nodo=color_por_nodo,
        grupo_por_nodo=liga_de_nodo,
        titulo=titulo or titulo_defecto,
        semilla=semilla,
        nodos_destacados=nodos_destacados,
    )
    return fig, aristas_f
