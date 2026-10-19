"""
Medidor de fichajes: precio máximo a ofrecer por un jugador para obtener
beneficio al revenderlo con la probabilidad elegida (ver `fichajes/modelo.py`).
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import estilo
from comun.banderas import url_bandera
from comun.datos import cargar_tabla, etiquetas_jugadores, perfil_jugadores
from prediccion import modelos as pm

from . import modelo as fm

CLAVE = "fich_"
DIAS_ACTIVO = 540          # jugadores y equipos con alguna valoración en el último año y medio
EQUIPO_POR_DEFECTO = "Real Madrid"
COLOR_BENEFICIO, COLOR_PERDIDA = "#10B981", "#EF4444"


# ---------------------------------------------------------------------------
# Datos y modelo (cacheados)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Analizando los traspasos históricos y ajustando el efecto traspaso...")
def modelo_traspasos(_df: pd.DataFrame, n_filas: int):
    anual = pm.serie_anual(_df)
    retornos = pm.retornos_anuales(anual)
    niveles = fm.Niveles(_df)
    params = fm.ParametrosPorNivel(retornos, niveles)
    traspasos = fm.detectar_traspasos(_df)
    con_dato = traspasos[traspasos["log_ret"].notna()].reset_index(drop=True)
    mu, sigma = fm.esperado(con_dato, params)
    con_dato["esperado"], con_dato["sigma_esperada"] = mu, sigma
    con_dato["residuo"] = con_dato["log_ret"] - mu
    # El efecto traspaso se ajusta con los traspasos recientes; la pestaña
    # "Traspasos similares" muestra todo el histórico.
    entreno = fm.entrenamiento(con_dato)
    modelo = fm.ajustar(entreno, entreno["residuo"].to_numpy(), entreno["sigma_esperada"].to_numpy(), niveles,
                        banda=params.banda_club(entreno["equipo_destino"], entreno["anio"]))
    etapa_actual = anual.groupby("id_player")["etapa"].last()
    return retornos, params, con_dato, modelo, etapa_actual


@st.cache_resource(show_spinner="Validando el efecto traspaso con los traspasos posteriores a 2018...")
def _validacion(_df: pd.DataFrame, _traspasos: pd.DataFrame, _retornos: pd.DataFrame, n_filas: int):
    return fm.validar(_df, _traspasos, _retornos)


@st.cache_data
def equipos_activos(_df: pd.DataFrame, n_filas: int) -> pd.DataFrame:
    """Equipos con valoraciones recientes: liga actual, nivel y plantilla."""
    recientes = _df[_df["fecha"] >= _df["fecha"].max() - pd.Timedelta(days=DIAS_ACTIVO)]
    ultimo = recientes.sort_values("fecha").groupby("equipo").tail(1).set_index("equipo")
    agregado = recientes.groupby("equipo").agg(valor_medio=("valor_mercado_eur", "mean"),
                                               jugadores=("id_player", "nunique"))
    return agregado.join(ultimo[["liga"]]).sort_values("valor_medio", ascending=False)


@st.cache_data(show_spinner="Simulando la carrera del jugador en su nuevo equipo...")
def _simular(v0: float, edad0: int, etapa0: int, anios: int, n: int, _params, clave_params: str,
             efecto: float, factor_sigma: float) -> np.ndarray:
    return fm.simular_fichaje(v0, edad0, etapa0, anios, _params, efecto, factor_sigma, n)


# ---------------------------------------------------------------------------
# Piezas de la página
# ---------------------------------------------------------------------------
def _bandera(pais) -> str:
    url = url_bandera(pais)
    return f'<img src="{url}" width="20">' if url else ""


ICONO_FIABILIDAD = {"habitual": "🟢", "poco": "🟡", "inusual": "🔴"}


def _ficha(p: pd.Series, destino: str, liga_destino: str, edad0: int, mu_grupo: float, efecto: float,
           banda: str, fiabilidad: str) -> None:
    variacion_sin = np.exp(mu_grupo) - 1
    variacion_con = np.exp(mu_grupo + efecto) - 1
    st.markdown(
        f"""<div class="ficha">
        <h3>{html.escape(str(p['nombre']))}</h3>
        <div class="dato">{_bandera(p['nacionalidad'])}{html.escape(str(p['nacionalidad']))} ·
        {html.escape(str(p['posicion'])) if pd.notna(p['posicion']) else 'Posición desconocida'}</div>
        <div class="dato">🎂 {edad0} años{f" · {int(p['altura_cm'])} cm" if pd.notna(p['altura_cm']) else ""}</div>
        <div class="dato" style="margin-top:0.6rem"><b>Traspaso</b></div>
        <div class="dato">{_bandera(p['liga_actual'])}{html.escape(str(p['equipo_actual']))}</div>
        <div class="dato">⬇️</div>
        <div class="dato">{_bandera(liga_destino)}<b>{html.escape(destino)}</b></div>
        <div class="dato" style="font-size:0.82rem;opacity:0.75">Comparado con clubes de su nivel: {html.escape(banda)}</div>
        <div class="dato" title="Qué parte de los traspasos que salen de su liga van a la liga de destino">
        {ICONO_FIABILIDAD[fiabilidad]} Movimiento entre ligas: <b>{fm.FIABILIDAD[fiabilidad].lower()}</b></div>
        <div class="dato" style="margin-top:0.6rem">Variación esperada el 1er año:</div>
        <div class="dato">· típica de un jugador como él en esos clubes: <b>{variacion_sin:+.0%}</b></div>
        <div class="dato">· fichando por {html.escape(destino)}: <b>{variacion_con:+.0%}</b></div>
        </div>""",
        unsafe_allow_html=True,
    )


def _grafico_curva(final: np.ndarray, v0: float, prob: float, coste: float, anios: int) -> go.Figure:
    probabilidades = fm.PROBABILIDADES
    ofertas = fm.oferta_maxima(final, probabilidades, coste, anios)
    oferta = float(fm.oferta_maxima(final, prob, coste, anios))
    fig = go.Figure()
    fig.add_scatter(x=probabilidades, y=ofertas, mode="lines", line=dict(color=estilo.COLOR_PRIMARIO, width=3),
                    fill="tozeroy", fillcolor="rgba(14,165,233,0.12)", name="Oferta máxima",
                    hovertemplate="Beneficio con prob. %{x:.0%}: pagar como mucho %{y:,.0f} €<extra></extra>")
    fig.add_hline(y=v0, line_dash="dash", line_color="#64748B",
                  annotation_text=f"Valor actual: {estilo.euros(v0)}", annotation_position="top right")
    fig.add_scatter(x=[prob], y=[oferta], mode="markers+text", marker=dict(size=14, color=estilo.COLOR_ACENTO,
                    line=dict(color="white", width=2)), text=[f"  {estilo.euros(oferta)}"], textposition="middle right",
                    textfont=dict(size=15, color=estilo.COLOR_PRIMARIO), showlegend=False, hoverinfo="skip")
    fig.update_layout(height=430, title="Oferta máxima según la probabilidad de beneficio exigida",
                      xaxis=dict(title="Probabilidad de obtener beneficio", tickformat=".0%"),
                      yaxis=dict(title="Oferta máxima (€)", rangemode="tozero"), showlegend=False,
                      margin=dict(t=50, l=10, r=10, b=10))
    return fig


def _pestana_reventa(final: np.ndarray, oferta: float, coste: float, anios: int, prob: float) -> None:
    umbral = oferta * (1 + coste) ** anios
    log_final = np.log10(final)
    bordes = np.linspace(log_final.min(), log_final.max(), 61)
    cuentas, _ = np.histogram(log_final, bordes)
    centros = (bordes[:-1] + bordes[1:]) / 2
    fig = go.Figure(go.Bar(
        x=centros, y=cuentas, width=np.diff(bordes),
        marker_color=np.where(centros >= np.log10(umbral), COLOR_BENEFICIO, COLOR_PERDIDA),
        customdata=10 ** centros, hovertemplate="~%{customdata:,.0f} €: %{y} simulaciones<extra></extra>"))
    fig.add_vline(x=np.log10(umbral), line_color="#0F172A", line_dash="dash",
                  annotation_text=f"Precio {'+ coste ' if coste else ''}: {estilo.euros(umbral)}")
    ticks = [v for v in [1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8]
             if bordes[0] - 0.2 <= np.log10(v) <= bordes[-1] + 0.2]
    fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[estilo.euros(v) for v in ticks],
                     title=f"Valor de mercado dentro de {anios} año(s) (escala log)")
    fig.update_layout(height=380, yaxis_title="Nº de simulaciones", bargap=0, showlegend=False,
                      margin=dict(t=30, l=10, r=10, b=10))
    c1, c2 = st.columns([2.2, 1], gap="large")
    c1.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Si pagas la oferta máxima**")
        ganancia = final / (1 + coste) ** anios - oferta
        st.dataframe(pd.DataFrame([
            ("Probabilidad de beneficio", f"{(ganancia > 0).mean():.0%}"),
            ("Beneficio mediano", estilo.euros(float(np.median(ganancia)))),
            ("Beneficio medio", estilo.euros(float(ganancia.mean()))),
            ("Pérdida media si sale mal", estilo.euros(float(ganancia[ganancia <= 0].mean())) if (ganancia <= 0).any() else "—"),
            ("Escenario pesimista (P5)", estilo.euros(float(np.percentile(final, 5)))),
            ("Escenario optimista (P95)", estilo.euros(float(np.percentile(final, 95)))),
        ], columns=["Métrica", "Valor"]), hide_index=True, use_container_width=True)
        st.caption(f"Verde: escenarios en los que el jugador vale más de lo pagado{' (más el coste de oportunidad)' if coste else ''} "
                   f"cuando se vende; son el {prob:.0%} de las simulaciones por construcción.")


ETIQUETAS_CORTAS = {
    "Traspaso medio": "Traspaso<br>medio", fm.PERFIL: "Perfil del<br>jugador", fm.NIVEL: "Nivel ligas<br>y clubes",
    fm.ADAPTACION: "Adaptación<br>al destino", fm.HIST_LIGA_DESTINO: "Historial<br>liga destino",
    fm.HIST_LIGAS: "Historial<br>ligas", fm.HIST_NACIONALIDAD: "Historial<br>nacionalidad",
    fm.HIST_DESTINO: "Historial<br>club destino", fm.HIST_CLUBES: "Historial<br>clubes",
}


def _pestana_efecto(prediccion: pd.Series, contrib: pd.Series, destino: str, componentes: list[str]) -> None:
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        valores = [float(prediccion[c]) for c in componentes]
        cortas = ETIQUETAS_CORTAS
        etiquetas = []
        for c in componentes:
            n = prediccion.get(f"n · {c}")
            etiquetas.append(f"{cortas[c]}<br><sub>{estilo.numero(n)} traspasos</sub>" if n is not None else cortas[c])
        fig = go.Figure(go.Waterfall(
            x=etiquetas + ["<b>Efecto<br>total</b>"], y=[v * 100 for v in valores] + [0],
            measure=["relative"] * len(valores) + ["total"],
            text=[f"{v * 100:+.1f}" for v in valores] + [f"{sum(valores) * 100:+.1f}"], textposition="outside",
            increasing=dict(marker_color=COLOR_BENEFICIO), decreasing=dict(marker_color=COLOR_PERDIDA),
            totals=dict(marker_color=estilo.COLOR_PRIMARIO), connector=dict(line=dict(color="#CBD5E1"))))
        fig.update_layout(height=430, title=f"Qué aporta el traspaso a {destino} a la variación del primer año",
                          yaxis_title="Puntos de variación (log × 100)", showlegend=False,
                          xaxis=dict(tickangle=0), margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Diferencia respecto a la variación esperada para un jugador de su posición, edad, año de carrera "
                   "y valor. Los historiales se «encogen» hacia 0 cuando hay pocos traspasos: un grupo con "
                   f"{fm.K_SUAVIZADO} traspasos cuenta la mitad que uno con muchísimos.")
    with c2:
        st.markdown("**Variables del perfil, del nivel y de la adaptación**")
        tabla = contrib[contrib.abs() > 1e-4].sort_values(key=np.abs, ascending=False).head(12)
        fig = go.Figure(go.Bar(x=tabla.to_numpy() * 100, y=tabla.index, orientation="h",
                               marker_color=np.where(tabla.to_numpy() >= 0, COLOR_BENEFICIO, COLOR_PERDIDA),
                               hovertemplate="%{y}: %{x:+.1f} puntos<extra></extra>"))
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"), xaxis_title="Puntos (log × 100)",
                          margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Aportación de cada variable para este jugador y este destino, respecto a un traspaso medio.")


def _pestana_similares(traspasos: pd.DataFrame, origen: str, destino: str, liga_origen: str, liga_destino: str) -> None:
    if fm.ajustes_version()["prueba_fiable"]:
        traspasos = traspasos[~(traspasos["cesion"] | traspasos["filial"])]
        st.caption("Solo traspasos fiables: sin cesiones (ida y vuelta al mismo club) ni cambios entre el filial y "
                   "el primer equipo.")
    opciones = {
        f"{origen} → {destino}": (traspasos["equipo_origen"] == origen) & (traspasos["equipo_destino"] == destino),
        f"Fichajes de {destino}": traspasos["equipo_destino"] == destino,
        f"Liga {liga_origen} → liga {liga_destino}": (traspasos["liga_origen"] == liga_origen)
                                                     & (traspasos["liga_destino"] == liga_destino),
    }
    etiquetas = {k: f"{k} ({estilo.numero(int(v.sum()))})" for k, v in opciones.items()}
    por_defecto = next((k for k, v in opciones.items() if v.sum() >= 5), list(opciones)[-1])
    grupo = st.segmented_control("Traspasos históricos", list(opciones), default=por_defecto,
                                 format_func=etiquetas.get, key=CLAVE + "similares") or por_defecto
    datos = traspasos[opciones[grupo]].sort_values("fecha", ascending=False)
    if datos.empty:
        st.info("No hay traspasos históricos con valoración un año después en este grupo.")
        return
    estilo.tarjetas_kpi([
        ("Traspasos", estilo.numero(len(datos)), f"{datos['fecha'].min():%Y} – {datos['fecha'].max():%Y}"),
        ("Variación real mediana (1 año)", f"{np.exp(datos['log_ret'].median()) - 1:+.0%}", "Anualizada"),
        ("Diferencia media vs. esperado", f"{datos['residuo'].mean() * 100:+.1f} puntos",
         "Positivo: rindieron por encima de lo esperado"),
        ("Revalorizados", f"{(datos['valor1'] > datos['valor0']).mean():.0%}", "Valían más un año después"),
    ])
    st.write("")
    mostrar = datos.head(300).assign(
        bandera=lambda d: d["nacionalidad"].map(lambda n: url_bandera(n) or None),
        variacion=lambda d: np.exp(d["log_ret"]) - 1,
        variacion_esperada=lambda d: np.exp(d["esperado"]) - 1,
    )
    st.dataframe(
        mostrar[["bandera", "nombre", "fecha", "equipo_origen", "equipo_destino", "edad", "posicion", "valor0",
                 "valor1", "variacion", "variacion_esperada"]],
        hide_index=True, use_container_width=True, height=420,
        column_config={
            "bandera": st.column_config.ImageColumn("", width="small"),
            "nombre": "Jugador", "fecha": st.column_config.DateColumn("Llegada", format="MM/YYYY"),
            "equipo_origen": "Origen", "equipo_destino": "Destino",
            "edad": st.column_config.NumberColumn("Edad", format="%d"), "posicion": "Posición",
            "valor0": st.column_config.NumberColumn("Valor antes", format="euro"),
            "valor1": st.column_config.NumberColumn("Valor 1 año después", format="euro"),
            "variacion": st.column_config.NumberColumn("Variación real (anual)", format="percent"),
            "variacion_esperada": st.column_config.NumberColumn("Variación esperada", format="percent"),
        },
    )


def _pestana_validacion(df: pd.DataFrame, traspasos: pd.DataFrame, retornos: pd.DataFrame) -> None:
    metricas, calibracion, info = _validacion(df, traspasos, retornos, len(traspasos))
    st.markdown(
        f"Backtest **fuera de muestra**: efecto traspaso ajustado solo con los traspasos que empiezan hasta "
        f"{info['anio_corte']} ({estilo.numero(info['n_train'])}) y evaluado, a 1 año, con los posteriores "
        f"({estilo.numero(info['n_test'])})."
    )
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.dataframe(metricas, hide_index=True, use_container_width=True, column_config={
            "modelo": "Modelo",
            "error_medio": st.column_config.NumberColumn("Error medio (log)", format="%.4f"),
            "nll": st.column_config.NumberColumn("NLL", format="%.4f",
                                                 help="Log-verosimilitud negativa: cuanto menor, mejor."),
            "correlacion": st.column_config.NumberColumn("Correlación", format="%.3f",
                                                         help="Entre el efecto previsto y el real."),
        })
        st.caption("Tener en cuenta el traspaso (ligas, clubes, perfil e historial) reduce el error y la NLL "
                   "respecto a usar solo el módulo de predicción. La correlación mide cuánto del «rendimiento "
                   "extra» real tras el traspaso anticipa el modelo.")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=[0.45, 1], y=[0.45, 1], mode="lines", line=dict(color="#94A3B8", dash="dot"),
                        name="Calibración perfecta")
        fig.add_scatter(x=calibracion["probabilidad_pedida"], y=calibracion["sin_efecto"], mode="lines+markers",
                        name="Sin efecto traspaso", line=dict(color="#F59E0B"))
        fig.add_scatter(x=calibracion["probabilidad_pedida"], y=calibracion["con_efecto"], mode="lines+markers",
                        name="Con efecto traspaso", line=dict(color=estilo.COLOR_PRIMARIO, width=3))
        fig.add_scatter(x=calibracion["probabilidad_pedida"], y=calibracion["con_efecto_elite"], mode="lines+markers",
                        name=f"Solo fichajes de clubes de élite ({estilo.numero(info['n_elite'])})",
                        line=dict(color="#EF4444", dash="dash"))
        fig.update_layout(height=360, title="¿Se obtiene beneficio con la probabilidad pedida?",
                          xaxis=dict(title="Probabilidad pedida", tickformat=".0%"),
                          yaxis=dict(title="Beneficio real (fichajes 2019+)", tickformat=".0%"),
                          legend=dict(orientation="h", y=-0.25), margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Para cada fichaje posterior a 2018 se calcula la oferta máxima a 1 año y se comprueba si el "
                   "jugador valía más un año después. Cuanto más cerca de la diagonal, más fiable es la probabilidad. "
                   f"En los clubes de élite ({info['elite']}) el modelo es algo optimista: tras 2018 los jugadores "
                   "caros cayeron más que en cualquier periodo anterior (COVID y ajuste del mercado), algo que no "
                   "podía aprenderse de los datos previos. Para esos clubes conviene exigir una probabilidad mayor.")


def _pestana_metodologia() -> None:
    st.markdown("#### 1. Traspasos históricos")
    st.markdown(
        "Cada cambio de club en el historial es un traspaso. Se toma el **último valor en el club de origen** y "
        "el valor **aproximadamente un año después** (entre 8 y 16 meses), y se calcula su variación anual."
    )
    st.markdown("#### 2. Efecto traspaso")
    st.markdown(
        "La variación real se compara con la **esperada** para un jugador de su posición, edad, año de carrera "
        "y valor (los parámetros del módulo de *Predicción*). Esa diferencia se explica con:\n"
        "- **Perfil del jugador**: edad, altura, posición, nacionalidad y si pasa a jugar en la liga de su país.\n"
        "- **Nivel de ligas y clubes**: diferencia de nivel entre liga y club de destino y de origen, y cuánto "
        "vale el jugador respecto al nivel del club destino. El nivel es el valor medio (geométrico) de los "
        f"jugadores de su **misma posición** en los **últimos {fm.VENTANA_NIVEL} años** (en los traspasos "
        "históricos, los 5 años que terminan en el del traspaso). Si un club tiene menos de "
        f"{fm.MIN_VALORACIONES_NIVEL} valoraciones de esa posición se usan todas sus posiciones, y si aun así "
        "no hay datos, todo su histórico.\n"
        "- **Historial**: cómo les fue a los jugadores que hicieron el mismo cambio de **liga**, a los que "
        "ficharon por el **club destino** y a los que hicieron el mismo cambio de **club**.\n\n"
        f"Las dos primeras partes son una regresión ridge (penalización {fm.LAMBDA_RIDGE:,.0f}, elegida fuera de "
        "muestra); las variables continuas se limitan al rango observado para no extrapolar. El historial usa la "
        f"media de cada grupo encogida hacia 0: efecto = suma / (n + {fm.K_SUAVIZADO}).".replace(",", ".")
    )
    st.markdown(
        f"Se ajusta solo con los traspasos de los **últimos {pm.VENTANA_ANIOS} años**, igual que los parámetros "
        "de la predicción: el mercado de hace 15-20 años crecía más que el actual."
    )
    if fm.ajustes_version()["nivel_asimetrico"]:
        st.markdown("#### Adaptabilidad (versión actual del modelo)")
        st.markdown(
            "- **Subir y bajar de nivel pesan distinto**: bajar a una liga o un club mucho más débil se penaliza "
            "por separado de subir.\n"
            "- **Traspasos poco habituales**: se mide qué parte de los traspasos que salen de la liga de origen "
            "van a la de destino (LaLiga → Premier es habitual; LaLiga → China, casi inédito). La ficha lo "
            f"muestra como habitual (≥ {fm.CUOTAS_FIABILIDAD[0]:.0%}), poco habitual o inusual "
            f"(< {fm.CUOTAS_FIABILIDAD[1]:.0%}).\n"
            "- **Más historiales**: el de todas las llegadas a la **liga destino** y el de los jugadores de su "
            "**nacionalidad** en esa liga, además de los de ligas y clubes.\n"
            "- Estas variables tienen menos penalización que el resto, para que pesen más.\n"
            "- La validación y la tabla de traspasos similares usan solo **traspasos fiables** (sin cesiones ni "
            "cambios con el filial).\n\n"
            "Fichar por un club de una liga mucho más débil y poco habitual (p. ej. un joven de LaLiga a China) "
            "reduce ahora bastante más el valor esperado que fichar por un club de su misma liga. Para volver al "
            "modelo anterior: `VERSION_MODELO = \"clasico\"` en `fichajes/modelo.py`."
        )
    st.markdown("#### 3. Grupo de referencia: clubes de nivel parecido")
    st.markdown(
        "La evolución esperada (μ y σ de su grupo) se estima con jugadores de su misma posición, edad y valor, "
        f"**en clubes de nivel parecido al club de destino**: los clubes se dividen en "
        f"{len(fm.CUANTILES_BANDA) + 1} bandas según el valor medio de su plantilla en los últimos 5 años (la más "
        "alta es la élite, ~5 % de los casos). Como en cada banda hay menos datos, se mezcla con la estimación "
        "general: la banda pesa n / (n + "
        f"{fm.K_BANDA}) según los casos que tenga de esa posición y edad. Validado fuera de muestra, esta mezcla "
        "mejora el modelo; separar del todo por bandas lo empeoraba (grupos demasiado pequeños)."
    )
    st.markdown("#### 4. Simulación y oferta máxima")
    st.latex(r"\ln V_{1} = \ln V_0 + (\mu_{grupo} + \text{efecto}) + f\,\sigma_{grupo} Z")
    st.markdown(
        "El primer año usa la tendencia de su grupo más el efecto traspaso y una volatilidad algo mayor (los "
        "años de traspaso son más inciertos); los siguientes, el Monte Carlo del módulo de predicción. Si "
        "$V_T$ es el valor al venderlo dentro de $T$ años y $c$ el coste de oportunidad anual:"
    )
    st.latex(r"P\big(V_T > \text{precio}\,(1+c)^T\big) = x \iff \text{precio} = \frac{P_{1-x}(V_T)}{(1+c)^T}")
    st.markdown("#### Limitaciones")
    st.markdown(
        "- El **valor de mercado** se usa como precio de reventa: la tabla no incluye importes reales de traspaso, "
        "salarios ni duración de contratos.\n"
        "- El efecto traspaso solo se aplica al primer año; después el jugador evoluciona como los de su grupo.\n"
        "- Es una estimación estadística del comportamiento típico, no una tasación individual."
    )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
def _inicial(clave: str, **por_defecto) -> dict:
    """Valor inicial de un widget, salvo que ya venga fijado en la sesión
    (p. ej. al abrir un jugador desde "Mejores fichajes"): pasar ambos haría
    que Streamlit mostrase un aviso."""
    return {} if CLAVE + clave in st.session_state else por_defecto


def render() -> None:
    df = cargar_tabla()
    perfiles = perfil_jugadores(df, float(len(df)))
    retornos, params, traspasos, modelo, etapa_actual = modelo_traspasos(df, len(df))
    equipos = equipos_activos(df, len(df))

    estilo.cabecera(
        "Medidor de fichajes",
        "¿Cuánto se puede pagar por un jugador para ganar dinero al revenderlo? Elige el equipo comprador, el "
        "jugador y la probabilidad de beneficio que exiges.",
    )

    lista_equipos = equipos.index.tolist()
    c_eq, c_jug = st.columns([1.4, 2.6], gap="large")
    destino = c_eq.selectbox(
        "Equipo comprador", lista_equipos,
        **_inicial("equipo", index=lista_equipos.index(EQUIPO_POR_DEFECTO) if EQUIPO_POR_DEFECTO in lista_equipos else 0),
        format_func=lambda e: f"{e} · {equipos.at[e, 'liga']}", key=CLAVE + "equipo",
        help="Equipos con valoraciones en el último año y medio, ordenados por valor medio de su plantilla.")
    liga_destino = str(equipos.at[destino, "liga"])

    activos = perfiles[perfiles["ultima_fecha"] >= df["fecha"].max() - pd.Timedelta(days=DIAS_ACTIVO)]
    ids = activos.sort_values("valor_actual", ascending=False).index.tolist()
    fuera = [i for i in ids if activos.at[i, "equipo_actual"] != destino]
    id_player = c_jug.selectbox(
        "Jugador", ids, **_inicial("jugador", index=ids.index(fuera[0]) if fuera else 0),
        format_func=etiquetas_jugadores(perfiles, float(len(perfiles)), True).get, key=CLAVE + "jugador",
        help="Jugadores en activo, ordenados por valor actual. Escribe para buscar.")

    c_prob, c_anios, c_coste, c_sim = st.columns([1.6, 1.2, 1.2, 1.2], gap="large")
    prob = c_prob.slider("Probabilidad de obtener beneficio", 50, 99, **_inicial("prob", value=70), format="%d %%",
                         key=CLAVE + "prob", help="Probabilidad de revenderlo por más de lo pagado.") / 100
    anios = c_anios.slider("Años hasta la venta", 1, 6, **_inicial("anios", value=3), key=CLAVE + "anios")
    coste = c_coste.slider("Coste de oportunidad anual", 0, 15, **_inicial("coste", value=0), format="%d %%",
                           key=CLAVE + "coste",
                           help="Rentabilidad anual que exiges al dinero invertido. Con 0 %, beneficio = "
                                "venderlo por más de lo pagado.") / 100
    n_sim = c_sim.select_slider("Nº de simulaciones", [2_000, 5_000, 10_000, 20_000], value=10_000,
                                format_func=estilo.numero, key=CLAVE + "n")

    p = perfiles.loc[id_player]
    if pd.isna(p["edad_actual"]):
        st.warning("La tabla de datos no incluye la edad de este jugador, necesaria para la simulación.")
        return
    edad0, etapa0, v0 = int(p["edad_actual"]), int(etapa_actual.loc[id_player]), float(p["valor_actual"])
    posicion = str(p["posicion"]) if p["posicion"] in fm.POSICIONES else None
    # Evolución tras el fichaje: jugadores de su posición en clubes del nivel del destino.
    banda = int(params.banda_club([destino], [params.niveles.anio_max])[0])
    params_jugador = params.para(banda, posicion)

    fila = pd.DataFrame([{
        "nacionalidad": p["nacionalidad"], "posicion": p["posicion"], "altura_cm": p["altura_cm"],
        "edad": float(edad0), "equipo_origen": p["equipo_actual"], "liga_origen": p["liga_actual"],
        "equipo_destino": destino, "liga_destino": liga_destino, "valor0": v0,
    }])
    ya_juega_alli = p["equipo_actual"] == destino
    prediccion = fm.predecir(modelo, fila).iloc[0]
    efecto = 0.0 if ya_juega_alli else float(prediccion["efecto"])
    factor = 1.0 if ya_juega_alli else modelo.factor(banda)
    a, e, t = (int(i[0]) for i in params_jugador.indices([edad0], [etapa0], [pm.tramo_de([v0])[0]]))
    mu_grupo = float(params_jugador.mu[a, e, t])

    if ya_juega_alli:
        st.info(f"{p['nombre']} ya juega en {destino}: se calcula sin efecto traspaso (renovación o recompra).")

    caminos = _simular(v0, edad0, etapa0, anios, n_sim, params_jugador, f"{banda}-{posicion}", efecto, factor)
    final = caminos[:, -1]
    oferta = float(fm.oferta_maxima(final, prob, coste, anios))
    prob_valor_actual = float((final > v0 * (1 + coste) ** anios).mean())

    st.divider()
    estilo.tarjetas_kpi([
        (f"Oferta máxima (beneficio {prob:.0%})", estilo.euros(oferta), f"{oferta / v0:.0%} de su valor actual"),
        ("Valor de mercado actual", estilo.euros(v0), f"{p['equipo_actual']} · {p['ultima_fecha']:%m/%Y}"),
        ("Prob. de beneficio pagando su valor", f"{prob_valor_actual:.0%}",
         "Si se paga exactamente su valor de mercado"),
        (f"Valor mediano al venderlo ({anios} año/s)", estilo.euros(float(np.median(final))),
         f"Rango 90 %: {estilo.euros(float(np.percentile(final, 5)))} – {estilo.euros(float(np.percentile(final, 95)))}"),
    ])
    st.write("")

    c_ficha, c_curva = st.columns([1, 3.2], gap="large")
    with c_ficha:
        _ficha(p, destino, liga_destino, edad0, mu_grupo, efecto, params.describir(banda),
               str(fm.fiabilidad(modelo.flujos, [p["liga_actual"]], [liga_destino])[0]))
        st.write("")
        veredicto = ("por encima" if oferta > v0 * 1.05 else "por debajo" if oferta < v0 * 0.95 else "en torno a")
        st.markdown(
            f"Para ganar dinero con una probabilidad del **{prob:.0%}** vendiéndolo dentro de {anios} año(s), "
            f"{html.escape(destino)} puede pagar como mucho **{estilo.euros(oferta)}**, {veredicto} de su valor "
            f"de mercado actual ({estilo.euros(v0)})."
        )
    with c_curva:
        st.plotly_chart(_grafico_curva(final, v0, prob, coste, anios), use_container_width=True)

    tabs = st.tabs(["💶 Valor de reventa", "🧩 Efecto del traspaso", "🔁 Traspasos similares", "✅ Validación",
                    "ℹ️ Cómo funciona"])
    with tabs[0]:
        _pestana_reventa(final, oferta, coste, anios, prob)
    with tabs[1]:
        if ya_juega_alli:
            st.info("Sin traspaso: el jugador ya pertenece al equipo comprador.")
        else:
            _pestana_efecto(prediccion, fm.contribuciones_perfil(modelo, fila), destino, fm.componentes(modelo))
    with tabs[2]:
        _pestana_similares(traspasos, str(p["equipo_actual"]), destino, str(p["liga_actual"]), liga_destino)
    with tabs[3]:
        _pestana_validacion(df, traspasos, retornos)
    with tabs[4]:
        _pestana_metodologia()
