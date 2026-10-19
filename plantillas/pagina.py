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
# Tres colores en todo el módulo: azul para la referencia (frontera y cartera
# eficiente), ámbar para el club elegido y gris para el resto de clubes.
AZUL, AZUL_CLARO, AMBAR, GRIS, GRIS_CLARO, TINTA = "#1E3A8A", "#93C5FD", "#F59E0B", "#94A3B8", "#E2E8F0", "#0F172A"
ESCALA_AZUL = ["#1E3A8A", "#1D4ED8", "#2563EB", "#3B82F6", "#60A5FA", "#7DB5FB", "#93C5FD", "#A9D1FD", "#BFDBFE",
               "#CFE4FE", "#DBEAFE", "#E6F0FE"]
MAX_PORCIONES = len(ESCALA_AZUL)      # jugadores con porción propia en el diagrama circular
FUENTE = "Inter, Segoe UI, sans-serif"
SIN_BARRA = {"displayModeBar": False}
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


def _base(fig: go.Figure, alto: int, titulo: str | None = None, **layout) -> go.Figure:
    """Aspecto común de las gráficas del módulo: fondo limpio, rejilla tenue
    y título discreto alineado a la izquierda."""
    fig.update_layout(height=alto, font=dict(family=FUENTE, size=13, color="#334155"), plot_bgcolor="white",
                      paper_bgcolor="white", margin=dict(t=56 if titulo else 16, l=10, r=16, b=10),
                      hoverlabel=dict(bgcolor="white", bordercolor=GRIS_CLARO, font=dict(family=FUENTE, color=TINTA)),
                      **layout)
    if titulo:
        fig.update_layout(title=dict(text=titulo, x=0, xanchor="left", font=dict(size=16, color=TINTA)))
    fig.update_xaxes(gridcolor="#F1F5F9", zeroline=False, linecolor=GRIS_CLARO, ticks="outside", tickcolor=GRIS_CLARO)
    fig.update_yaxes(gridcolor="#F1F5F9", zeroline=False, linecolor=GRIS_CLARO)
    return fig


# ---------------------------------------------------------------------------
# Diagrama circular
# ---------------------------------------------------------------------------
def _diagrama_circular(plantilla: pd.DataFrame, club: pd.Series, equipo: str, alto: int = 500) -> None:
    orden = plantilla.sort_values("valor", ascending=False)
    cabeza, cola = orden.head(MAX_PORCIONES), orden.iloc[MAX_PORCIONES:]
    etiquetas, valores = cabeza["nombre"].tolist(), cabeza["valor"].tolist()
    detalle = [f"{d} · {int(e)} años<br>{estilo.euros(v)} · {_pct(w, False)} de la plantilla"
               for d, e, v, w in zip(cabeza["demarcacion"], cabeza["edad"], cabeza["valor"], cabeza["peso"])]
    colores = ESCALA_AZUL[:len(cabeza)]
    if len(cola):
        etiquetas.append(f"Resto ({len(cola)} jugadores)"); valores.append(float(cola["valor"].sum()))
        detalle.append(f"Jugadores de menor valor<br>{estilo.euros(float(cola['valor'].sum()))} · "
                       f"{_pct(float(cola['peso'].sum()), False)} de la plantilla")
        colores = colores + [GRIS_CLARO]
    pesos = np.array(valores) / sum(valores)
    centro = f"<b>{estilo.euros(club['valor'])}</b><br><span style='font-size:12px;color:#64748B'>{int(club['jugadores'])} jugadores</span>"
    fig = go.Figure(go.Pie(
        labels=etiquetas, values=valores, customdata=detalle, hole=0.58, sort=False, direction="clockwise",
        marker=dict(colors=colores, line=dict(color="white", width=2)),
        text=[f"{e}<br>{_pct(p, False)}" if p >= 0.045 else "" for e, p in zip(etiquetas, pesos)], textinfo="text",
        textposition="outside",
        textfont=dict(size=12, color="#334155"), outsidetextfont=dict(size=12, color="#334155"),
        hovertemplate="<b>%{label}</b><br>%{customdata}<extra></extra>"))
    _base(fig, alto, f"Reparto del valor de mercado de {equipo}", showlegend=False,
          annotations=[dict(text=centro, showarrow=False, font=dict(size=22, color=TINTA, family=FUENTE))])
    fig.update_layout(margin=dict(t=64, l=30, r=30, b=30))
    st.plotly_chart(fig, use_container_width=True, config=SIN_BARRA)


# ---------------------------------------------------------------------------
# Piezas de la página
# ---------------------------------------------------------------------------
def _barras_reparto(club: pd.Series, clubes: pd.DataFrame, titulo: str, columnas: list[str], nombres: list[str]) -> go.Figure:
    media = clubes[columnas].mean().to_numpy(dtype=float)
    valores = club[columnas].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_bar(x=valores, y=nombres, orientation="h", marker_color=AZUL, width=0.55, name="Club",
                text=[_pct(v, False) for v in valores], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x:.1%}<extra></extra>")
    fig.add_scatter(x=media, y=nombres, mode="markers", name="Media de los clubes",
                    marker=dict(symbol="line-ns", size=20, line=dict(color=GRIS, width=3)),
                    hovertemplate="Media de los clubes: %{x:.1%}<extra></extra>")
    _base(fig, 190, None, showlegend=False, xaxis=dict(tickformat=".0%", range=[0, max(0.6, float(valores.max()) * 1.22)]),
          yaxis=dict(autorange="reversed"))
    fig.update_layout(margin=dict(t=30, l=10, r=40, b=5),
                      annotations=[dict(text=f"<b>{titulo}</b>", x=0, y=1.18, xref="paper", yref="paper", showarrow=False,
                                        font=dict(size=13, color=TINTA))])
    return fig


def _pestana_reparto(plantilla: pd.DataFrame, club: pd.Series, clubes: pd.DataFrame, mercado: pm.Mercado, equipo: str) -> None:
    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        _diagrama_circular(plantilla, club, equipo)
        st.caption("Pasa el cursor por una porción para ver la demarcación, la edad, el valor y el peso del jugador.")
    with c2:
        orden = plantilla.sort_values("valor", ascending=False)
        st.markdown("**¿Depende de pocos jugadores?**")
        st.markdown(
            f"- El jugador más valioso supone el **{_pct(club['mayor_peso'], False)}** del valor de la plantilla, y "
            f"los cinco primeros el **{_pct(float(orden['peso'].head(5).sum()), False)}**.\n"
            f"- Tiene {int(club['jugadores'])} jugadores, pero su valor está repartido como si fueran "
            f"**{club['jugadores_efectivos']:.1f}** de igual valor (jugadores efectivos).\n"
            f"- Esa concentración aporta **{_puntos(club['riesgo_propio'])}** puntos de riesgo propio, frente a "
            f"**{_puntos(club['riesgo_mercado'])}** de riesgo de mercado."
        )
        st.plotly_chart(_barras_reparto(club, clubes, "Reparto por líneas", [f"linea_{l}" for l in pm.LINEAS],
                                        list(pm.LINEAS_ES.values())), use_container_width=True, config=SIN_BARRA)
        st.plotly_chart(_barras_reparto(club, clubes, "Reparto por grupos de edad",
                                        [f"edad_{j}" for j in range(len(pm.EDADES))], pm.EDADES),
                        use_container_width=True, config=SIN_BARRA)
        st.caption("La marca gris es la media de los 96 clubes.")

    st.markdown("**Jugadores de la plantilla**")
    tabla = plantilla.assign(
        bandera=plantilla["nacionalidad"].map(lambda n: url_bandera(n) or None),
        rentabilidad=mercado.mu[plantilla["segmento"].to_numpy()],
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
    resto = clubes[clubes.index != equipo]
    de_su_liga = resto["liga"] == club["liga"]
    # Encuadre: la zona en la que están los clubes, no toda la curva.
    x0, x1 = float(clubes["riesgo"].min()) - 0.012, float(clubes["riesgo"].max()) + 0.012
    y0 = float(clubes["rentabilidad"].min()) - 0.04
    y1 = float(np.interp(x1, curva["riesgo"], curva["rentabilidad"])) + 0.03
    fig = go.Figure()
    # Zona por encima de la frontera: ninguna cartera llega hasta ahí.
    fig.add_scatter(x=[*curva["riesgo"], x1 + 1, x1 + 1, curva["riesgo"].iloc[0]],
                    y=[*curva["rentabilidad"], curva["rentabilidad"].iloc[-1], 2, 2], mode="none", fill="toself",
                    fillcolor="rgba(30,58,138,0.045)", hoverinfo="skip", showlegend=False)
    for nombre, filtro, color, tam in [("Otros clubes", ~de_su_liga, "#CBD5E1", 9),
                                       (f"Clubes de {LIGAS_ES[club['liga']]}", de_su_liga, GRIS, 10)]:
        d = resto[filtro]
        fig.add_scatter(x=d["riesgo"], y=d["rentabilidad"], mode="markers", name=nombre, text=d.index,
                        customdata=np.column_stack([d["puesto"], d["brecha"] * 100]),
                        marker=dict(size=tam, color=color, line=dict(color="white", width=1),
                                    opacity=np.where(d["liga"].isin(ligas), 1.0, 0.35)),
                        hovertemplate="<b>%{text}</b><br>Riesgo %{x:.1%} · rentabilidad %{y:+.1%}<br>"
                                      "Puesto %{customdata[0]} · a %{customdata[1]:.1f} puntos de la frontera<extra></extra>")
    fig.add_scatter(x=curva["riesgo"], y=curva["rentabilidad"], mode="lines", name="Frontera eficiente",
                    line=dict(color=AZUL, width=3.5, shape="spline"),
                    hovertemplate="Frontera: riesgo %{x:.1%} · rentabilidad %{y:+.1%}<extra></extra>")
    fig.add_scatter(x=[eficiente["riesgo"]], y=[eficiente["rentabilidad"]], mode="markers",
                    name="Cartera eficiente con su mismo riesgo",
                    marker=dict(size=13, color="white", symbol="circle", line=dict(color=AZUL, width=3)),
                    hovertemplate="Cartera eficiente: riesgo %{x:.1%} · rentabilidad %{y:+.1%}<extra></extra>")
    fig.add_scatter(x=[club["riesgo"]], y=[club["rentabilidad"]], mode="markers", name=equipo,
                    marker=dict(size=18, color=AMBAR, symbol="circle", line=dict(color="white", width=2.5)),
                    hovertemplate=f"<b>{html.escape(equipo)}</b><br>Riesgo %{{x:.1%}} · rentabilidad %{{y:+.1%}}<extra></extra>")
    # La brecha: flecha del club a la cartera eficiente de su mismo riesgo.
    fig.add_annotation(x=eficiente["riesgo"], y=eficiente["rentabilidad"], ax=club["riesgo"], ay=club["rentabilidad"],
                       xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=2,
                       arrowcolor=AMBAR, standoff=9, startstandoff=12, text="")
    fig.add_annotation(x=club["riesgo"], y=(club["rentabilidad"] + eficiente["rentabilidad"]) / 2, xanchor="left", xshift=10,
                       text=f"<b>{_puntos(club['brecha'])} puntos</b><br>de brecha", showarrow=False, align="left",
                       font=dict(size=12, color="#B45309"))
    fig.add_annotation(x=club["riesgo"], y=club["rentabilidad"], yshift=-22, showarrow=False,
                       text=f"<b>{html.escape(equipo)}</b>", font=dict(size=13, color=TINTA))
    fig.add_annotation(x=x0 + 0.004, y=y1 - 0.012, xanchor="left", yanchor="top", showarrow=False, align="left",
                       text="Zona inalcanzable:<br>más rentabilidad exige más riesgo", font=dict(size=11, color=GRIS))
    _base(fig, 580, "Frontera eficiente y posición de las 96 plantillas",
          xaxis=dict(title="Riesgo (volatilidad anual del valor de la plantilla)", tickformat=".0%", range=[x0, x1], dtick=0.02),
          yaxis=dict(title="Rentabilidad esperada (variación anual del valor)", tickformat="+.0%", range=[y0, y1], dtick=0.1,
                     zeroline=True, zerolinecolor=GRIS_CLARO, zerolinewidth=1.5),
          legend=dict(orientation="h", y=-0.17, x=0, font=dict(size=12)), hovermode="closest")
    st.plotly_chart(fig, use_container_width=True, config=SIN_BARRA)
    st.caption(
        "Cada punto es una plantilla; pasa el cursor para ver el club. La curva es la frontera eficiente: la mayor "
        f"rentabilidad esperada que se puede conseguir con cada nivel de riesgo, repartiendo el valor por igual entre "
        f"{pm.N_FRONTERA} jugadores. Cuanto más cerca de la curva, mejor diversificada está la plantilla. A "
        f"{equipo} le separan **{_puntos(club['brecha'])} puntos** de la cartera eficiente de su mismo riesgo: "
        f"**{_puntos(club['brecha_concentracion'])}** por la concentración del valor en pocos jugadores y "
        f"**{_puntos(club['brecha_reparto'])}** por el reparto entre líneas y edades.")


def _pestana_comparacion(club: pd.Series, clubes: pd.DataFrame, curva: pd.DataFrame, equipo: str) -> None:
    eficiente = pm.cartera_eficiente(curva, club["riesgo"])
    W_ef = eficiente[list(range(len(pm.SEGMENTOS)))].to_numpy(dtype=float).reshape(len(pm.LINEAS), len(pm.EDADES))
    W = club[[f"seg_{k}" for k in range(len(pm.SEGMENTOS))]].to_numpy(dtype=float).reshape(W_ef.shape)
    media = clubes[[f"seg_{k}" for k in range(len(pm.SEGMENTOS))]].mean().to_numpy().reshape(W_ef.shape)
    c1, c2 = st.columns(2, gap="large")
    for columna, titulo, eje, nombres, leyenda in [(c1, "Por grupos de edad", 0, pm.EDADES, True),
                                                  (c2, "Por líneas", 1, list(pm.LINEAS_ES.values()), False)]:
        fig = go.Figure()
        for nombre, pesos, color in [(equipo, W, AMBAR), ("Cartera eficiente (mismo riesgo)", W_ef, AZUL),
                                     ("Media de los 96 clubes", media, GRIS_CLARO)]:
            v = pesos.sum(axis=eje)
            fig.add_bar(x=nombres, y=v, name=nombre, marker_color=color, text=[f"{x:.0%}" for x in v],
                        textposition="outside", textfont=dict(size=11, color="#475569"), cliponaxis=False,
                        hovertemplate="%{x}: %{y:.1%}<extra>" + html.escape(nombre) + "</extra>")
        _base(fig, 400, titulo, barmode="group", bargap=0.28, bargroupgap=0.08, showlegend=leyenda,
              yaxis=dict(title="Parte del valor", tickformat=".0%", rangemode="tozero"),
              legend=dict(orientation="h", y=-0.14, x=0, font=dict(size=12)))
        columna.plotly_chart(fig, use_container_width=True, config=SIN_BARRA)

    diferencia = W_ef - W
    c3, c4 = st.columns([1.3, 1], gap="large")
    with c3:
        limite = float(np.abs(diferencia).max())
        fig = go.Figure(go.Heatmap(
            z=diferencia, x=pm.EDADES, y=list(pm.LINEAS_ES.values()), zmid=0, zmin=-limite, zmax=limite,
            colorscale=[[0, AMBAR], [0.5, "#F8FAFC"], [1, AZUL]], xgap=4, ygap=4,
            text=[[("" if abs(v) < 0.005 else f"{v * 100:+.0f}".replace("-", "−")) for v in fila] for fila in diferencia],
            texttemplate="%{text}", textfont=dict(size=14),
            colorbar=dict(title="Puntos", tickformat="+.0%", thickness=12, outlinewidth=0),
            hovertemplate="%{y} · %{x}: %{z:+.1%}<extra></extra>"))
        _base(fig, 340, "Qué cambiaría: cartera eficiente menos plantilla actual", yaxis=dict(autorange="reversed"))
        fig.update_xaxes(showgrid=False, ticks=""); fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True, config=SIN_BARRA)
        st.caption("Puntos de peso que la cartera eficiente tiene de más (azul) o de menos (ámbar) que la plantilla.")
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
        _pestana_reparto(plantilla, club, clubes, mercado, equipo)
    with tabs[1]:
        _pestana_frontera(clubes, curva, equipo, ligas)
    with tabs[2]:
        _pestana_comparacion(club, clubes, curva, equipo)
    with tabs[3]:
        _pestana_clubes(clubes, equipo, ligas)
    with tabs[4]:
        _pestana_metodologia(mercado, limites)
