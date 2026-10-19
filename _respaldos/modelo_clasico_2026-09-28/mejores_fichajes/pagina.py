"""
Mejores fichajes: ranking de los jugadores más rentables que puede fichar un
equipo en una posición y rango de precio (ver `mejores_fichajes/calculo.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import estilo
from comun.banderas import url_bandera
from comun.datos import cargar_tabla, perfil_jugadores
from comun.navegacion import PAGINAS
from fichajes import modelo as fm
from fichajes.pagina import DIAS_ACTIVO, EQUIPO_POR_DEFECTO, equipos_activos, modelo_traspasos

from . import calculo as mc

CLAVE = "mejf_"
PRECIOS = [0, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000,
           15_000_000, 20_000_000, 30_000_000, 50_000_000, 75_000_000, 100_000_000, 150_000_000, 200_000_000]
N_PRIMERA = 1_000      # simulaciones por candidato en la criba
N_REFINADO = 10_000    # simulaciones de los que se muestran (mismas que el Medidor de fichajes)
ORIGEN = {"todas": "Cualquier liga", "misma": "Solo de su liga", "otras": "Solo de otras ligas"}


# ---------------------------------------------------------------------------
# Cálculo (cacheado)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Simulando la carrera de los candidatos en el nuevo equipo...", max_entries=12)
def _evaluar(ids: tuple, destino: str, liga_destino: str, anios: int, n: int, _candidatos: pd.DataFrame,
             _modelo, _params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return mc.evaluar(_candidatos, _modelo, _params, destino, liga_destino, anios, n)


def _candidatos(perfiles: pd.DataFrame, etapa: pd.Series, fecha_max: pd.Timestamp, destino: str) -> pd.DataFrame:
    """Jugadores en activo (con edad) que no juegan ya en el equipo comprador."""
    activos = perfiles[(perfiles["ultima_fecha"] >= fecha_max - pd.Timedelta(days=DIAS_ACTIVO))
                       & perfiles["edad_actual"].notna() & (perfiles["equipo_actual"] != destino)]
    return activos.assign(etapa=etapa.reindex(activos.index).to_numpy(),
                          edad_actual=activos["edad_actual"].astype(float))


# ---------------------------------------------------------------------------
# Piezas de la página
# ---------------------------------------------------------------------------
def _filtros_categorias(candidatos: pd.DataFrame, liga_destino: str) -> pd.Series:
    """Filtros por categorías (desplegable). Devuelve la máscara de candidatos."""
    with st.expander("🏷️ Filtros por categorías", expanded=False):
        c1, c2, c3 = st.columns(3, gap="large")
        edad = c1.slider("Edad", 16, 40, (16, 40), key=CLAVE + "edad")
        altura = c2.slider("Altura (cm)", 150, 210, (150, 210), key=CLAVE + "altura")
        sin_altura = c2.checkbox("Incluir jugadores sin altura conocida", value=True, key=CLAVE + "sin_altura")
        pies = c3.multiselect("Pie dominante", sorted(candidatos["pie"].dropna().unique()), key=CLAVE + "pie",
                              placeholder="Todos")
        c4, c5, c6 = st.columns(3, gap="large")
        nacionalidades = c4.multiselect("Nacionalidad", sorted(candidatos["nacionalidad"].dropna().unique()),
                                        key=CLAVE + "nacionalidad", placeholder="Todas")
        ligas = c5.multiselect("Liga actual", sorted(candidatos["liga_actual"].dropna().unique()),
                               key=CLAVE + "liga", placeholder="Todas")
        origen = c6.segmented_control("Procedencia", list(ORIGEN), format_func=ORIGEN.get, default="todas",
                                      key=CLAVE + "origen") or "todas"

    mascara = candidatos["edad_actual"].between(*edad)
    altura_ok = candidatos["altura_cm"].between(*altura)
    mascara &= altura_ok | (candidatos["altura_cm"].isna() & sin_altura)
    if pies:
        mascara &= candidatos["pie"].isin(pies)
    if nacionalidades:
        mascara &= candidatos["nacionalidad"].isin(nacionalidades)
    if ligas:
        mascara &= candidatos["liga_actual"].isin(ligas)
    if origen == "misma":
        mascara &= candidatos["liga_actual"] == liga_destino
    elif origen == "otras":
        mascara &= candidatos["liga_actual"] != liga_destino
    return mascara.fillna(False).astype(bool)


def _tabla(top: pd.DataFrame, prob: float, anios: int) -> dict:
    mostrar = top.assign(
        bandera=top["nacionalidad"].map(lambda n: url_bandera(n) or None),
        posicion_es=top["posicion"].map(mc.POSICIONES_ES),
    ).reset_index()
    columnas = ["bandera", "nombre", "edad_actual", "posicion_es", "altura_cm", "pie", "equipo_actual",
                "liga_actual", "precio", "beneficio_esperado", "beneficio_mediano", "beneficio_asegurado",
                "oferta_maxima", "prob_beneficio", "margen", "rentabilidad_mediana", "efecto"]
    return st.dataframe(
        mostrar[columnas], hide_index=True, use_container_width=True, height=min(38 + 35 * len(mostrar), 740),
        on_select="rerun", selection_mode="single-row", key=CLAVE + "tabla",
        column_config={
            "bandera": st.column_config.ImageColumn("", width="small"),
            "nombre": "Jugador",
            "edad_actual": st.column_config.NumberColumn("Edad", format="%d"),
            "posicion_es": "Posición",
            "altura_cm": st.column_config.NumberColumn("Altura", format="%d cm"),
            "pie": "Pie",
            "equipo_actual": "Equipo actual",
            "liga_actual": "Liga",
            "precio": st.column_config.NumberColumn("Precio", format="compact",
                                                    help="Valor de mercado actual más la prima elegida (€)."),
            "oferta_maxima": st.column_config.NumberColumn(
                f"Oferta máx. ({prob:.0%})", format="compact",
                help=f"Lo máximo que se puede pagar para obtener beneficio con probabilidad {prob:.0%} (€)."),
            "margen": st.column_config.NumberColumn(
                f"Margen al {prob:.0%}", format="percent",
                help=f"Oferta máxima / precio − 1: con probabilidad {prob:.0%}, la rentabilidad es al menos esta."),
            "rentabilidad_mediana": st.column_config.NumberColumn(
                f"Rentab. mediana ({anios} a)", format="percent",
                help="Rentabilidad en el escenario central al venderlo."),
            "beneficio_esperado": st.column_config.NumberColumn(
                "Beneficio esperado (€)", format="compact",
                help="Dinero que genera de media: valor medio de venta − precio (€)."),
            "beneficio_mediano": st.column_config.NumberColumn(
                "Beneficio mediano (€)", format="compact",
                help="Dinero que genera en el escenario central: valor mediano de venta − precio (€)."),
            "beneficio_asegurado": st.column_config.NumberColumn(
                f"Beneficio al {prob:.0%} (€)", format="compact",
                help=f"Con probabilidad {prob:.0%}, gana al menos esto: oferta máxima − precio (€). "
                     "Negativo: con esa probabilidad no se garantiza beneficio."),
            "prob_beneficio": st.column_config.ProgressColumn(
                "Prob. de beneficio", format="percent", min_value=0.0, max_value=1.0,
                help="Probabilidad de venderlo por más de lo pagado (más el coste de oportunidad)."),
            "efecto": st.column_config.NumberColumn(
                "Efecto traspaso", format="percent",
                help="Variación extra del primer año por fichar por este equipo (Medidor de fichajes)."),
        },
    )


def _grafico(resultados: pd.DataFrame, top: pd.DataFrame, prob: float, anios: int) -> go.Figure:
    """Dinero que genera cada candidato frente a la probabilidad de ganarlo."""
    muestra = resultados.sample(min(len(resultados), 4000), random_state=0)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=muestra["prob_beneficio"], y=muestra["beneficio_esperado"], mode="markers", name="Candidatos",
        marker=dict(size=6, color="rgba(100,116,139,0.28)"), customdata=muestra[["nombre", "equipo_actual"]],
        hovertemplate="%{customdata[0]} (%{customdata[1]})<br>Prob. beneficio %{x:.0%} · beneficio esperado "
                      "%{y:,.0f} €<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=top["prob_beneficio"], y=top["beneficio_esperado"], mode="markers+text", name="Ranking",
        text=top["nombre"].head(10).tolist() + [""] * max(len(top) - 10, 0), textposition="top center",
        textfont=dict(size=11), marker=dict(size=11, color=top["beneficio_asegurado"], colorscale="RdYlGn", cmid=0,
                                            line=dict(color="white", width=1),
                                            colorbar=dict(title=f"Beneficio<br>al {prob:.0%} (€)")),
        customdata=top[["nombre", "equipo_actual", "beneficio_asegurado", "precio"]],
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Precio %{customdata[3]:,.0f} €<br>"
                      "Prob. beneficio %{x:.0%} · beneficio esperado %{y:,.0f} €<br>"
                      f"Beneficio al {prob:.0%}: " + "%{customdata[2]:,.0f} €<extra></extra>"))
    fig.add_vline(x=prob, line_dash="dot", line_color="#0F172A", annotation_text=f"Prob. exigida {prob:.0%}")
    fig.add_hline(y=0, line_color="#CBD5E1")
    fig.update_layout(height=520, title="Dinero que genera cada candidato y probabilidad de ganarlo",
                      xaxis=dict(title="Probabilidad de beneficio pagando su precio", tickformat=".0%"),
                      yaxis=dict(title=f"Beneficio esperado al venderlo en {anios} año(s) (€)"),
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=50, l=10, r=10, b=10))
    return fig


def _metodologia() -> None:
    st.markdown("#### Cómo se elige a los mejores")
    st.markdown(
        "1. **Candidatos**: jugadores en activo (valoración en el último año y medio) de la posición elegida, "
        "que no juegan ya en el equipo y cuyo **precio** está en el rango. El precio es su valor de mercado más "
        "la **prima** elegida (la tabla no incluye importes reales de traspaso).\n"
        "2. **Efecto traspaso** (módulo *Medidor de fichajes*): cuánto sube o baja su valor el primer año por "
        "fichar por este equipo, según su perfil, el nivel reciente de ligas y clubes en su posición y el "
        "historial de traspasos parecidos.\n"
        "3. **Simulación** (módulo *Predicción*): Monte Carlo de su carrera por posición, edad, año de carrera "
        "y tramo de valor, con el efecto traspaso en el primer año.\n"
        f"4. **Criba y refinado**: todos los candidatos se simulan {estilo.numero(N_PRIMERA)} veces para "
        f"ordenarlos; los que aparecen en la tabla se vuelven a simular {estilo.numero(N_REFINADO)} veces (como "
        "en el Medidor) y se reordenan. Así las cifras mostradas son precisas y se evita que un jugador suba "
        "en el ranking solo por azar de la simulación."
    )
    st.markdown("#### Métricas")
    st.markdown(
        "Por defecto el ranking se ordena por el **dinero que genera** cada fichaje, en euros (no en "
        "porcentaje): ganar 10 M€ con un jugador de 50 M€ cuenta más que ganar 3 M€ con uno de 1 M€.\n\n"
        "- **Beneficio esperado (€)**: valor medio de venta − precio. Es el dinero que genera de media; incluye "
        "los escenarios muy buenos, por lo que favorece a jugadores con mucho recorrido al alza.\n"
        "- **Beneficio mediano (€)**: el del escenario central (la mitad de las veces gana más, la otra mitad "
        "menos). Más prudente que el esperado.\n"
        "- **Beneficio con la probabilidad elegida (€)**: oferta máxima − precio. Con esa probabilidad se gana "
        "al menos esto; si es negativo, no se garantiza beneficio con esa probabilidad.\n"
        "- **Oferta máxima**: lo máximo que se puede pagar para ganar dinero con la probabilidad elegida.\n"
        "- **Margen (%)** y **rentabilidad mediana (%)**: las mismas ideas en proporción al precio "
        "(favorecen a los jugadores baratos).\n"
        "- **Probabilidad de beneficio**: de venderlo por más de lo pagado (más el coste de oportunidad)."
    )
    st.markdown("#### Limitaciones")
    st.markdown(
        "- El valor de mercado se usa como precio de compra y de venta; no hay salarios ni duración de contratos.\n"
        "- Los modelos reproducen el comportamiento **típico**: los jugadores muy jóvenes y baratos suelen "
        "salir muy rentables porque, de media, su valor sube mucho; su riesgo individual (lesiones, estancarse) "
        "es mayor de lo que parece. Usa los filtros por categorías para acotar el perfil buscado."
    )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
def render() -> None:
    df = cargar_tabla()
    perfiles = perfil_jugadores(df, float(len(df)))
    _, params, _, modelo, etapa_actual = modelo_traspasos(df, len(df))
    equipos = equipos_activos(df, len(df))

    estilo.cabecera(
        "Mejores fichajes",
        "Los jugadores más rentables que puede fichar un equipo: elige posición y presupuesto y compara cuánto "
        "ganaría al revenderlos.",
    )

    lista_equipos = equipos.index.tolist()
    c_eq, c_pos, c_precio = st.columns([1.3, 1.9, 1.8], gap="large")
    destino = c_eq.selectbox(
        "Equipo comprador", lista_equipos,
        index=lista_equipos.index(EQUIPO_POR_DEFECTO) if EQUIPO_POR_DEFECTO in lista_equipos else 0,
        format_func=lambda e: f"{e} · {equipos.at[e, 'liga']}", key=CLAVE + "equipo")
    liga_destino = str(equipos.at[destino, "liga"])
    opciones_pos = ["todas", *fm.POSICIONES]
    posicion = c_pos.segmented_control(
        "Posición", opciones_pos, default="Attack", key=CLAVE + "posicion",
        format_func=lambda p: "Todas" if p == "todas" else mc.POSICIONES_ES[p]) or "todas"
    precio_min, precio_max = c_precio.select_slider(
        "Precio a pagar (mínimo – máximo)", PRECIOS, value=(5_000_000, 30_000_000), format_func=estilo.euros,
        key=CLAVE + "precio")

    c_prob, c_anios, c_prima, c_coste = st.columns(4, gap="large")
    prob = c_prob.slider("Probabilidad de beneficio exigida", 50, 99, 70, format="%d %%", key=CLAVE + "prob") / 100
    anios = c_anios.slider("Años hasta la venta", 1, 6, 3, key=CLAVE + "anios")
    prima = c_prima.slider("Prima sobre el valor de mercado", 0, 100, 0, step=5, format="%d %%", key=CLAVE + "prima",
                           help="Lo que se paga por encima del valor de mercado (cláusulas, competencia...).") / 100
    coste = c_coste.slider("Coste de oportunidad anual", 0, 15, 0, format="%d %%", key=CLAVE + "coste") / 100

    candidatos = _candidatos(perfiles, etapa_actual, df["fecha"].max(), destino)
    if posicion != "todas":
        candidatos = candidatos[candidatos["posicion"] == posicion]
    precio = candidatos["valor_actual"] * (1 + prima)
    candidatos = candidatos[(precio >= precio_min) & (precio <= precio_max)]
    candidatos = candidatos[_filtros_categorias(candidatos, liga_destino)]

    c_crit, c_n = st.columns([2, 1], gap="large")
    criterio = c_crit.selectbox("Ordenar por", list(mc.CRITERIOS), format_func=mc.CRITERIOS.get,
                                key=CLAVE + "criterio")
    n_mostrar = c_n.slider("Jugadores en el ranking", 10, 100, 25, step=5, key=CLAVE + "n")

    if candidatos.empty:
        st.info("Ningún jugador cumple esos criterios. Amplía el rango de precio o los filtros.")
        return

    # 1) Criba: todos los candidatos con pocas simulaciones.
    efecto, rejilla, media = _evaluar(tuple(candidatos.index), destino, liga_destino, anios, N_PRIMERA,
                                      candidatos, modelo, params)
    resultados = candidatos.join(
        mc.metricas(candidatos["valor_actual"], rejilla, media, prima, coste, anios, prob).set_index(candidatos.index)
    ).assign(efecto=np.exp(efecto) - 1)
    # 2) Refinado: los mejores de la criba (con holgura) con más simulaciones, y se reordenan.
    finalistas = resultados.nlargest(min(len(resultados), 2 * n_mostrar), criterio)
    ef_f, rej_f, med_f = _evaluar(tuple(finalistas.index), destino, liga_destino, anios, N_REFINADO,
                                  candidatos.loc[finalistas.index], modelo, params)
    finalistas = candidatos.loc[finalistas.index].join(
        mc.metricas(finalistas["valor_actual"], rej_f, med_f, prima, coste, anios, prob).set_index(finalistas.index)
    ).assign(efecto=np.exp(ef_f) - 1)
    top = finalistas.nlargest(n_mostrar, criterio)

    st.divider()
    mejor = top.iloc[0]
    estilo.tarjetas_kpi([
        ("Candidatos analizados", estilo.numero(len(candidatos)),
         f"{'Todas las posiciones' if posicion == 'todas' else mc.POSICIONES_ES[posicion]} · "
         f"{estilo.euros(precio_min)} – {estilo.euros(precio_max)}"),
        (f"Rentables al {prob:.0%}", estilo.numero(int((resultados["margen"] > 0).sum())),
         f"Oferta máxima por encima del precio ({(resultados['margen'] > 0).mean():.0%})"),
        ("Mejor oportunidad", str(mejor["nombre"]),
         f"{mc.CRITERIOS[criterio]}: " + (estilo.euros(mejor[criterio]) if criterio in mc.CRITERIOS_EUROS
                                         else f"{mejor[criterio]:+.0%}" if criterio != "prob_beneficio"
                                         else f"{mejor[criterio]:.0%}")),
        (f"Beneficio esperado del top {len(top)}", estilo.euros(float(top["beneficio_esperado"].sum())),
         f"Invirtiendo {estilo.euros(float(top['precio'].sum()))} en todos ellos"),
    ])
    st.write("")

    tabs = st.tabs(["📋 Ranking", "📈 Dinero y riesgo", "ℹ️ Cómo funciona"])
    with tabs[0]:
        seleccion = _tabla(top, prob, anios)
        filas = seleccion.selection.rows if seleccion else []
        if filas:
            id_player = int(top.index[filas[0]])
            nombre = top.loc[id_player, "nombre"]
            if st.button(f"🔍 Analizar a {nombre} en el Medidor de fichajes", type="primary", key=CLAVE + "abrir"):
                st.session_state.update({
                    "fich_equipo": destino, "fich_jugador": id_player, "fich_prob": int(round(prob * 100)),
                    "fich_anios": anios, "fich_coste": int(round(coste * 100)),
                })
                st.switch_page(PAGINAS["fichajes"])
        else:
            st.caption("Selecciona un jugador (casilla de la izquierda) para analizarlo en detalle en el "
                       "Medidor de fichajes.")
        st.caption(f"Precio = valor de mercado{f' + {prima:.0%} de prima' if prima else ''}. Venta dentro de "
                   f"{anios} año(s){f' con un coste de oportunidad del {coste:.0%} anual' if coste else ''}.")
    with tabs[1]:
        st.plotly_chart(_grafico(resultados, top, prob, anios), use_container_width=True)
        st.caption("Cada punto es un candidato (criba). Arriba a la derecha, los que más ganan con menos riesgo; "
                   "a la derecha de la línea punteada, los que dan beneficio con al menos la probabilidad exigida.")
    with tabs[2]:
        _metodologia()
