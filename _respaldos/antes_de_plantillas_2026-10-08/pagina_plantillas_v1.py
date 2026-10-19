"""
Gestión de plantillas: la plantilla de cada club como una cartera de
inversión, con la frontera eficiente de Markowitz (ver `plantillas/modelo.py`).
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import estilo
from comun.banderas import url_bandera
from comun.datos import cargar_tabla

from . import modelo as pm

CLAVE = "plan_"
EQUIPO_POR_DEFECTO = "Real Madrid"
LIGAS_ES = {"England": "Inglaterra", "Spain": "España", "Italy": "Italia", "Germany": "Alemania", "France": "Francia"}
COLOR_LIGA = dict(zip(pm.LIGAS, ["#8B5CF6", "#F59E0B", "#10B981", "#64748B", "#0EA5E9"]))
COLOR_CLUB, COLOR_EFICIENTE, COLOR_MEDIA = "#EF4444", "#10B981", "#94A3B8"
MAX_PORCIONES = 12      # jugadores con porción propia en el diagrama circular
VEREDICTOS = [(1 / 3, "está bien diversificada", "🟢"), (2 / 3, "tiene una diversificación media", "🟡"),
              (1.01, "está poco diversificada", "🔴")]


# ---------------------------------------------------------------------------
# Datos y modelo (cacheados)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Estimando el mercado y calculando la frontera eficiente...")
def estudio(_df: pd.DataFrame, n_filas: int, marca_plantillas: float):
    """Plantillas, mercado, clubes y frontera. `marca_plantillas` invalida la
    caché cuando se sustituye el fichero de plantillas."""
    plantillas = pm.cargar_plantillas()
    mercado = pm.estimar_mercado(pm.retornos_historicos(_df))
    clubes = pm.analizar_clubes(plantillas, mercado)
    limites = pm.limites_linea(clubes)
    curva = pm.frontera(mercado, limites)
    return plantillas, mercado, pm.comparar_con_frontera(clubes, curva, mercado), limites, curva


def _pct(x: float, signo: bool = True) -> str:
    return (f"{x:+.1%}" if signo else f"{x:.1%}").replace(".", ",").replace("-", "−")


def _puntos(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",").replace("-", "−")


# ---------------------------------------------------------------------------
# Piezas de la página
# ---------------------------------------------------------------------------
def _pestana_reparto(plantilla: pd.DataFrame, club: pd.Series, mercado: pm.Mercado, equipo: str) -> None:
    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        orden = plantilla.sort_values("valor", ascending=False)
        cabeza, cola = orden.head(MAX_PORCIONES), orden.iloc[MAX_PORCIONES:]
        etiquetas, valores = cabeza["nombre"].tolist(), cabeza["valor"].tolist()
        if len(cola):
            etiquetas.append(f"Resto ({len(cola)} jugadores)"); valores.append(float(cola["valor"].sum()))
        colores = (estilo.PALETA * 2)[:len(cabeza)] + ["#CBD5E1"]
        fig = go.Figure(go.Pie(labels=etiquetas, values=valores, hole=0.45, sort=False, direction="clockwise",
                               marker=dict(colors=colores, line=dict(color="white", width=2)),
                               textinfo="label+percent", textposition="outside",
                               hovertemplate="%{label}: %{value:,.0f} € (%{percent})<extra></extra>"))
        fig.update_layout(height=470, showlegend=False, title=f"Reparto del valor de mercado de {equipo}",
                          margin=dict(t=60, l=10, r=10, b=10),
                          annotations=[dict(text=f"<b>{estilo.euros(club['valor'])}</b>", showarrow=False, font_size=17)])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**¿Depende de pocos jugadores?**")
        st.markdown(
            f"- El jugador más valioso supone el **{_pct(club['mayor_peso'], False)}** del valor de la plantilla, y "
            f"los cinco primeros el **{_pct(float(orden['peso'].head(5).sum()), False)}**.\n"
            f"- Tiene {int(club['jugadores'])} jugadores, pero su valor está repartido como si fueran "
            f"**{club['jugadores_efectivos']:.1f}** de igual valor (jugadores efectivos).\n"
            f"- Esa concentración aporta **{_puntos(club['riesgo_propio'])}** puntos de riesgo propio, frente a "
            f"**{_puntos(club['riesgo_mercado'])}** de riesgo de mercado."
        )
        for titulo, columnas, nombres in [("Por líneas", [f"linea_{l}" for l in pm.LINEAS], list(pm.LINEAS_ES.values())),
                                          ("Por grupos de edad", [f"edad_{j}" for j in range(len(pm.EDADES))], pm.EDADES)]:
            fig = go.Figure(go.Bar(x=club[columnas].to_numpy(dtype=float), y=nombres, orientation="h",
                                   marker_color=estilo.COLOR_PRIMARIO,
                                   text=[_pct(v, False) for v in club[columnas]], textposition="outside",
                                   hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig.update_layout(height=175, title=dict(text=titulo, font_size=14), margin=dict(t=35, l=10, r=40, b=5),
                              xaxis=dict(tickformat=".0%", range=[0, max(0.6, float(club[columnas].max()) * 1.2)]),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Jugadores de la plantilla**")
    tabla = plantilla.assign(
        bandera=plantilla["nacionalidad"].map(lambda n: url_bandera(n) or None),
        linea=plantilla["posicion"].map(pm.LINEAS_ES), rentabilidad=mercado.mu[plantilla["segmento"].to_numpy()],
        aportacion=pm.aportacion_al_riesgo(plantilla, mercado),
    ).sort_values("valor", ascending=False)
    st.dataframe(
        tabla[["bandera", "nombre", "demarcacion", "edad", "valor", "peso", "rentabilidad", "aportacion"]],
        hide_index=True, use_container_width=True, height=420,
        column_config={
            "bandera": st.column_config.ImageColumn("", width="small"), "nombre": "Jugador", "demarcacion": "Demarcación",
            "edad": st.column_config.NumberColumn("Edad", format="%d"),
            "valor": st.column_config.NumberColumn("Valor de mercado", format="euro"),
            "peso": st.column_config.ProgressColumn("Peso en la plantilla", format="percent", min_value=0.0,
                                                    max_value=float(tabla["peso"].max())),
            "rentabilidad": st.column_config.NumberColumn(
                "Rentab. esperada", format="percent",
                help="Variación anual media del valor de los jugadores de su línea y grupo de edad."),
            "aportacion": st.column_config.NumberColumn(
                "Aporta al riesgo", format="percent",
                help="Parte del riesgo (varianza) de la plantilla que se debe a este jugador."),
        },
    )


def _pestana_frontera(clubes: pd.DataFrame, curva: pd.DataFrame, equipo: str, ligas: list[str]) -> None:
    club = clubes.loc[equipo]
    eficiente = pm.cartera_eficiente(curva, club["riesgo"])
    fig = go.Figure()
    fig.add_scatter(x=curva["riesgo"], y=curva["rentabilidad"], mode="lines", name="Frontera eficiente",
                    line=dict(color=estilo.COLOR_PRIMARIO, width=3),
                    hovertemplate="Frontera: riesgo %{x:.1%}, rentabilidad %{y:+.1%}<extra></extra>")
    for liga in pm.LIGAS:
        d = clubes[(clubes["liga"] == liga) & (clubes.index != equipo)]
        fig.add_scatter(x=d["riesgo"], y=d["rentabilidad"], mode="markers", name=LIGAS_ES[liga], text=d.index,
                        marker=dict(size=9, color=COLOR_LIGA[liga], opacity=0.85 if liga in ligas else 0.15,
                                    line=dict(color="white", width=1)),
                        hovertemplate="<b>%{text}</b><br>Riesgo %{x:.1%} · rentabilidad %{y:+.1%}<extra></extra>")
    fig.add_scatter(x=[club["riesgo"], eficiente["riesgo"]], y=[club["rentabilidad"], eficiente["rentabilidad"]],
                    mode="lines", line=dict(color=COLOR_CLUB, dash="dot", width=2), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=[eficiente["riesgo"]], y=[eficiente["rentabilidad"]], mode="markers",
                    name="Cartera eficiente con su mismo riesgo",
                    marker=dict(size=13, color=COLOR_EFICIENTE, symbol="diamond", line=dict(color="white", width=1.5)),
                    hovertemplate="Cartera eficiente: riesgo %{x:.1%}, rentabilidad %{y:+.1%}<extra></extra>")
    fig.add_scatter(x=[club["riesgo"]], y=[club["rentabilidad"]], mode="markers+text", name=equipo, text=[f"<b>{equipo}</b>"],
                    textposition="bottom center", textfont=dict(size=14, color=COLOR_CLUB),
                    marker=dict(size=20, color=COLOR_CLUB, symbol="star", line=dict(color="white", width=1.5)),
                    hovertemplate=f"<b>{html.escape(equipo)}</b><br>Riesgo %{{x:.1%}} · rentabilidad %{{y:+.1%}}<extra></extra>")
    fig.update_layout(height=560, title="Frontera eficiente y posición de las 96 plantillas",
                      xaxis=dict(title="Riesgo (volatilidad anual del valor de la plantilla)", tickformat=".0%"),
                      yaxis=dict(title="Rentabilidad esperada (variación anual del valor)", tickformat="+.0%"),
                      legend=dict(orientation="h", y=-0.16), margin=dict(t=50, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Cada punto es una plantilla. La curva es la frontera eficiente: la mayor rentabilidad esperada que se puede "
        f"conseguir con cada nivel de riesgo, repartiendo el valor por igual entre {pm.N_FRONTERA} jugadores. "
        f"Cuanto más cerca de la curva, mejor diversificada está la plantilla. La línea punteada une a {equipo} con "
        f"la cartera eficiente de su mismo riesgo: le separan **{_puntos(club['brecha'])} puntos** de rentabilidad, "
        f"**{_puntos(club['brecha_concentracion'])}** por la concentración del valor en pocos jugadores y "
        f"**{_puntos(club['brecha_reparto'])}** por el reparto entre líneas y edades.")


def _pestana_comparacion(club: pd.Series, clubes: pd.DataFrame, curva: pd.DataFrame, equipo: str) -> None:
    eficiente = pm.cartera_eficiente(curva, club["riesgo"])
    W_ef = eficiente[list(range(len(pm.SEGMENTOS)))].to_numpy(dtype=float).reshape(len(pm.LINEAS), len(pm.EDADES))
    W = club[[f"seg_{k}" for k in range(len(pm.SEGMENTOS))]].to_numpy(dtype=float).reshape(W_ef.shape)
    media = clubes[[f"seg_{k}" for k in range(len(pm.SEGMENTOS))]].mean().to_numpy().reshape(W_ef.shape)
    c1, c2 = st.columns(2, gap="large")
    for columna, titulo, eje, nombres in [(c1, "Por grupos de edad", 0, pm.EDADES),
                                         (c2, "Por líneas", 1, list(pm.LINEAS_ES.values()))]:
        fig = go.Figure()
        for nombre, pesos, color in [(equipo, W, COLOR_CLUB), ("Cartera eficiente (mismo riesgo)", W_ef, COLOR_EFICIENTE),
                                     ("Media de los 96 clubes", media, COLOR_MEDIA)]:
            fig.add_bar(x=nombres, y=pesos.sum(axis=eje), name=nombre, marker_color=color,
                        hovertemplate="%{x}: %{y:.1%}<extra>" + html.escape(nombre) + "</extra>")
        fig.update_layout(height=380, title=titulo, barmode="group", yaxis=dict(title="Parte del valor", tickformat=".0%"),
                          legend=dict(orientation="h", y=-0.18), margin=dict(t=50, l=10, r=10, b=10))
        columna.plotly_chart(fig, use_container_width=True)

    diferencia = W_ef - W
    c3, c4 = st.columns([1.3, 1], gap="large")
    with c3:
        limite = float(np.abs(diferencia).max())
        fig = go.Figure(go.Heatmap(
            z=diferencia, x=pm.EDADES, y=list(pm.LINEAS_ES.values()), zmid=0, zmin=-limite, zmax=limite,
            colorscale=[[0, COLOR_CLUB], [0.5, "#F8FAFC"], [1, COLOR_EFICIENTE]], xgap=3, ygap=3,
            text=[[f"{v * 100:+.0f}".replace("-", "−") for v in fila] for fila in diferencia], texttemplate="%{text}",
            colorbar=dict(title="Puntos", tickformat="+.0%"), hovertemplate="%{y} · %{x}: %{z:+.1%}<extra></extra>"))
        fig.update_layout(height=330, title="Cartera eficiente menos plantilla actual (puntos de peso)",
                          yaxis=dict(autorange="reversed"), margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.markdown("**Cómo se acercaría a la frontera**")
        dif_edad, dif_linea = diferencia.sum(axis=0), diferencia.sum(axis=1)
        subir, bajar = int(np.argmax(dif_edad)), int(np.argmin(dif_edad))
        st.markdown(
            f"Con su mismo riesgo ({_pct(club['riesgo'], False)}), la cartera eficiente espera una variación del valor "
            f"de **{_pct(eficiente['rentabilidad'])}** al año, frente al **{_pct(club['rentabilidad'])}** de "
            f"{html.escape(equipo)}. Para acercarse a ella:\n"
            f"- **Más peso** en jugadores de {pm.EDADES[subir].lower()} ({_puntos(dif_edad[subir])} puntos más).\n"
            f"- **Menos peso** en jugadores de {pm.EDADES[bajar].lower()} ({_puntos(-dif_edad[bajar])} puntos menos).\n"
            f"- Por líneas, el mayor ajuste es en {list(pm.LINEAS_ES.values())[int(np.argmax(np.abs(dif_linea)))].lower()} "
            f"({_puntos(dif_linea[int(np.argmax(np.abs(dif_linea)))])} puntos).\n"
            f"- Repartir el valor entre más jugadores: tiene {club['jugadores_efectivos']:.1f} jugadores efectivos y la "
            f"frontera supone {pm.N_FRONTERA}."
        )
        st.caption("La cartera eficiente es una referencia financiera: maximiza la revalorización esperada de la "
                   "plantilla para ese riesgo, sin tener en cuenta el rendimiento deportivo.")


def _pestana_clubes(clubes: pd.DataFrame, equipo: str, ligas: list[str]) -> None:
    tabla = clubes[clubes["liga"].isin(ligas)].sort_values("puesto").reset_index()
    tabla = tabla.assign(bandera=tabla["liga"].map(lambda l: url_bandera(l) or None),
                         elegido=np.where(tabla["equipo"] == equipo, "⭐", ""))
    st.dataframe(
        tabla[["puesto", "elegido", "bandera", "equipo", "valor", "rentabilidad", "riesgo", "brecha", "brecha_reparto",
               "brecha_concentracion", "jugadores_efectivos", "mayor_peso", "edad_media"]],
        hide_index=True, use_container_width=True, height=560,
        column_config={
            "puesto": st.column_config.NumberColumn("Puesto", help="1 = la plantilla más cercana a la frontera."),
            "elegido": st.column_config.TextColumn("", width="small"),
            "bandera": st.column_config.ImageColumn("", width="small"), "equipo": "Club",
            "valor": st.column_config.NumberColumn("Valor de la plantilla", format="compact"),
            "rentabilidad": st.column_config.NumberColumn("Rentab. esperada", format="percent"),
            "riesgo": st.column_config.NumberColumn("Riesgo", format="percent"),
            "brecha": st.column_config.NumberColumn(
                "Brecha con la frontera", format="percent",
                help="Rentabilidad que le falta respecto a la cartera eficiente de su mismo riesgo."),
            "brecha_reparto": st.column_config.NumberColumn("… por líneas y edades", format="percent"),
            "brecha_concentracion": st.column_config.NumberColumn("… por concentración", format="percent"),
            "jugadores_efectivos": st.column_config.NumberColumn("Jugadores efectivos", format="%.1f"),
            "mayor_peso": st.column_config.NumberColumn("Mayor peso", format="percent"),
            "edad_media": st.column_config.NumberColumn("Edad media (por valor)", format="%.1f"),
        },
    )
    st.caption("Ordenados de la plantilla más cercana a la frontera eficiente a la más lejana. La brecha se separa "
               "en la parte que se debe al reparto del valor entre líneas y edades y la que se debe a su "
               "concentración en pocos jugadores.")


def _pestana_metodologia(mercado: pm.Mercado, limites: np.ndarray) -> None:
    st.markdown("#### 1. La plantilla como cartera")
    st.markdown(
        "Cada jugador es un activo y su **peso** es la parte del valor de mercado de la plantilla que representa. La "
        "**rentabilidad** de la cartera es la variación anual de ese valor, y el **riesgo**, su volatilidad.")
    st.latex(r"w_i = \frac{V_i}{\sum_j V_j} \qquad R_p = \sum_i w_i R_i")
    st.markdown("#### 2. Segmentos y mercado")
    st.markdown(
        f"Los jugadores se agrupan en {len(pm.SEGMENTOS)} segmentos (línea × grupo de edad). Con el histórico de las "
        f"cinco grandes ligas desde {pm.ANIO_MIN} se construye para cada segmento un índice ponderado por valor: la "
        "variación anual del valor conjunto de sus jugadores. De esos índices salen la rentabilidad esperada de cada "
        f"segmento (media de los últimos {pm.VENTANA_MU} años), su volatilidad y la covarianza entre segmentos "
        f"(correlación media: {mercado.correlacion_media:.2f}). La variación de un jugador es la de su segmento más "
        "una parte propia, independiente de las demás.")
    segmentos = pd.DataFrame({
        "Línea": [pm.LINEAS_ES[l] for l, _ in pm.SEGMENTOS], "Edad": [pm.EDADES[e] for _, e in pm.SEGMENTOS],
        "Rentabilidad esperada": mercado.mu, "Volatilidad del índice": mercado.sigma,
        "Riesgo propio de un jugador": np.sqrt(mercado.propio), "Variaciones históricas": mercado.n})
    st.dataframe(segmentos, hide_index=True, use_container_width=True, height=38 + 35 * len(segmentos), column_config={
        "Rentabilidad esperada": st.column_config.NumberColumn(format="percent"),
        "Volatilidad del índice": st.column_config.NumberColumn(format="percent"),
        "Riesgo propio de un jugador": st.column_config.NumberColumn(format="percent"),
        "Variaciones históricas": st.column_config.NumberColumn(format="localized")})
    st.markdown("#### 3. Rentabilidad y riesgo de una plantilla")
    st.latex(r"\mu_p = \sum_s W_s\,\mu_s \qquad \sigma_p^2 = W^{T}\Sigma\,W + \sum_i w_i^2\,d_{s(i)}")
    st.markdown(
        "donde $W_s$ es la parte del valor que hay en cada segmento, $\\Sigma$ la covarianza entre segmentos y "
        "$d_s$ la varianza propia de un jugador del segmento. El primer sumando es el **riesgo de mercado**, que se "
        "reduce combinando segmentos poco correlacionados; el segundo, el **riesgo propio**, que se reduce "
        "repartiendo el valor entre más jugadores.")
    st.markdown("#### 4. Frontera eficiente")
    st.markdown(
        "Para cada rentabilidad se busca el reparto por segmentos de menor riesgo (optimización cuadrática), con el "
        f"valor repartido por igual entre {pm.N_FRONTERA} jugadores y cada línea dentro del rango en el que se mueve "
        "el 90 % de los clubes: " + ", ".join(
            f"{pm.LINEAS_ES[l].lower()} entre el {limites[i, 0]:.0%} y el {limites[i, 1]:.0%}" for i, l in enumerate(pm.LINEAS))
        + ". La **brecha** de un club es la rentabilidad que le falta respecto a la cartera eficiente de su mismo riesgo.")
    st.markdown("#### Limitaciones")
    st.markdown(
        "- Solo se mide el valor de mercado: una plantilla eficiente lo es como inversión, no necesariamente en lo "
        "deportivo. Los jugadores veteranos pierden valor, pero pueden ser los que ganan partidos.\n"
        "- A igual línea y edad se supone la misma rentabilidad esperada, sea cual sea el valor del jugador.\n"
        "- La rentabilidad y el riesgo son los del pasado; nada garantiza que se repitan.")


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
def render() -> None:
    estilo.cabecera(
        "Gestión de plantillas",
        "La plantilla de cada club estudiada como una cartera de inversión: cómo reparte su valor de mercado y si "
        "está bien diversificada según la frontera eficiente de Markowitz.",
    )
    if not pm.ARCHIVO.exists():
        st.error(f"No se encontró `{pm.ARCHIVO.name}` en la carpeta `datos/`.")
        return
    df = cargar_tabla()
    plantillas, mercado, clubes, limites, curva = estudio(df, len(df), pm.ARCHIVO.stat().st_mtime)

    c_liga, c_club = st.columns([1.6, 2.4], gap="large")
    ligas = c_liga.pills("Ligas", pm.LIGAS, selection_mode="multi", default=pm.LIGAS, format_func=LIGAS_ES.get,
                         key=CLAVE + "ligas") or pm.LIGAS
    equipos = clubes[clubes["liga"].isin(ligas)].sort_values("valor", ascending=False).index.tolist()
    equipo = c_club.selectbox(
        "Club", equipos, index=equipos.index(EQUIPO_POR_DEFECTO) if EQUIPO_POR_DEFECTO in equipos else 0,
        format_func=lambda e: f"{e} · {LIGAS_ES[clubes.at[e, 'liga']]} ({estilo.euros(clubes.at[e, 'valor'])})",
        key=CLAVE + "equipo", help="Clubes de las cinco grandes ligas, ordenados por el valor de su plantilla.")
    club, plantilla = clubes.loc[equipo], plantillas[plantillas["equipo"] == equipo]

    estilo.tarjetas_kpi([
        ("Valor de la plantilla", estilo.euros(club["valor"]), f"{int(club['jugadores'])} jugadores"),
        ("Rentabilidad esperada", _pct(club["rentabilidad"]), "Variación anual del valor"),
        ("Riesgo", _pct(club["riesgo"], False),
         f"Mercado {_puntos(club['riesgo_mercado'])} · propio {_puntos(club['riesgo_propio'])}"),
        ("Jugadores efectivos", f"{club['jugadores_efectivos']:.1f}".replace(".", ","),
         f"El más valioso pesa el {_pct(club['mayor_peso'], False)}"),
        ("Diversificación", f"{int(club['puesto'])}º de {len(clubes)}", f"A {_puntos(club['brecha'])} puntos de la frontera"),
    ])
    veredicto, icono = next((t, i) for limite, t, i in VEREDICTOS if club["puesto"] / len(clubes) <= limite)
    causa = ("sobre todo por la concentración del valor en pocos jugadores"
             if club["brecha_concentracion"] > club["brecha_reparto"] else "sobre todo por su reparto entre líneas y edades")
    st.markdown(
        f"{icono} La plantilla de **{html.escape(equipo)}** **{veredicto}**: es la {int(club['puesto'])}ª de "
        f"{len(clubes)} más cercana a la frontera eficiente. Con su mismo riesgo, una cartera eficiente esperaría "
        f"{_puntos(club['brecha'])} puntos más de rentabilidad al año, {causa}.")

    tabs = st.tabs(["🥧 Reparto del valor", "📈 Frontera eficiente", "⚖️ Plantilla frente a cartera eficiente",
                    "🏟️ Todos los clubes", "ℹ️ Cómo funciona"])
    with tabs[0]:
        _pestana_reparto(plantilla, club, mercado, equipo)
    with tabs[1]:
        _pestana_frontera(clubes, curva, equipo, ligas)
    with tabs[2]:
        _pestana_comparacion(club, clubes, curva, equipo)
    with tabs[3]:
        _pestana_clubes(clubes, equipo, ligas)
    with tabs[4]:
        _pestana_metodologia(mercado, limites)
