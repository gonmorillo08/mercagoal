"""
Predicción de valores de mercado: simulación de Monte Carlo de la carrera
de un jugador durante los próximos años (ver `prediccion/modelos.py`).
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from comun import estilo
from comun.banderas import url_bandera
from comun.datos import cargar_tabla, etiquetas_jugadores, perfil_jugadores

from . import modelos as m

CLAVE = "pred_"
MODELO_MC = "Monte Carlo por edad y etapa de carrera"
MODELO_BOOT = "Simulación histórica (bootstrap)"
MODELOS = {
    MODELO_MC: "Movimiento Browniano Geométrico cuya tendencia y volatilidad dependen, cada año simulado, de la "
        "edad del jugador, su año de carrera y el nivel de valor de esa trayectoria (estimadas con los jugadores "
        "de su misma posición). Es el modelo recomendado.",
    MODELO_BOOT: "Cada año, cada trayectoria toma al azar una variación REAL de otro jugador en su misma "
        "situación (posición, edad, año de carrera y nivel de valor). No supone que las variaciones sean normales, así "
        "que recoge los saltos bruscos que se dan en la realidad.",
}
BANDA_EXTERIOR = "rgba(14,165,233,0.16)"
BANDA_INTERIOR = "rgba(14,165,233,0.32)"


# ---------------------------------------------------------------------------
# Datos y parámetros (cacheados)
# ---------------------------------------------------------------------------
# cache_resource (no cache_data): las series anuales (~270.000 filas) se
# comparten sin copiarse en cada recarga; nunca se modifican.
@st.cache_resource(show_spinner="Preparando las series anuales...")
def _datos(_df: pd.DataFrame, n_filas: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    anual = m.serie_anual(_df)
    return anual, m.retornos_anuales(anual)


@st.cache_resource(show_spinner="Estimando tendencia y volatilidad por edad y etapa de carrera...")
def _parametros(_retornos: pd.DataFrame, n_filas: int, posicion: str | None) -> m.ParametrosCarrera:
    return m.ParametrosCarrera(_retornos, posicion)


@st.cache_resource(show_spinner="Validando los modelos con datos posteriores a 2018...")
def _validacion(_retornos: pd.DataFrame, n_filas: int) -> pd.DataFrame:
    return m.validar(_retornos)


@st.cache_data(show_spinner="Simulando trayectorias...")
def _simular(modelo: str, v0: float, edad0: int, etapa0: int, anios: int, n: int, _params,
             clave_params: str) -> np.ndarray:
    if modelo == MODELO_BOOT:
        return m.simular_bootstrap(v0, edad0, etapa0, anios, _params, n)
    return m.simular_montecarlo(v0, edad0, etapa0, anios, _params, n)


# ---------------------------------------------------------------------------
# Piezas de la página
# ---------------------------------------------------------------------------
def _ficha(p: pd.Series, edad0: int, etapa0: int) -> None:
    bandera = url_bandera(p["nacionalidad"])
    fisico = "".join([
        f" · {int(p['altura_cm'])} cm" if pd.notna(p["altura_cm"]) else "",
        f" · pie {html.escape(str(p['pie']).lower())}" if pd.notna(p["pie"]) else "",
    ])
    st.markdown(
        f"""<div class="ficha">
        <h3>{html.escape(str(p['nombre']))}</h3>
        <div class="dato">{f'<img src="{bandera}" width="22">' if bandera else ''}{html.escape(str(p['nacionalidad']))}
        · {html.escape(str(p['posicion'])) if pd.notna(p['posicion']) else 'Posición desconocida'}</div>
        <div class="dato">🏟️ {html.escape(str(p['equipo_actual']))} ({html.escape(str(p['liga_actual']))})</div>
        <div class="dato">🎂 Edad: <b>{edad0} años</b>{fisico}</div>
        <div class="dato">📅 Año de carrera: <b>{etapa0 + 1}</b> (primera valoración en {p['primera_fecha']:%Y})</div>
        <div class="dato">🏆 Máximo: {estilo.euros(p['valor_pico'])} ({p['fecha_pico']:%m/%Y})</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _grafico_abanico(historial: pd.DataFrame, caminos: np.ndarray, fecha0: pd.Timestamp, log: bool,
                     ver_caminos: bool) -> go.Figure:
    pct = m.percentiles_por_anio(caminos)
    fechas = [fecha0 + pd.DateOffset(years=int(a)) for a in pct["anio"]]
    fig = go.Figure()
    if ver_caminos:
        rng = np.random.default_rng(1)
        for i in rng.choice(caminos.shape[0], size=min(30, caminos.shape[0]), replace=False):
            fig.add_scatter(x=fechas, y=caminos[i], mode="lines", line=dict(color="rgba(100,116,139,0.25)", width=1),
                            hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=fechas, y=pct["p95"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=fechas, y=pct["p5"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=BANDA_EXTERIOR,
                    name="Intervalo 90 % (P5–P95)", hoverinfo="skip")
    fig.add_scatter(x=fechas, y=pct["p75"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=fechas, y=pct["p25"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=BANDA_INTERIOR,
                    name="Intervalo 50 % (P25–P75)", hoverinfo="skip")
    fig.add_scatter(x=fechas, y=pct["p50"], mode="lines+markers", line=dict(color=estilo.COLOR_ACENTO, width=3, dash="dash"),
                    name="Mediana simulada", customdata=np.stack([pct["p5"], pct["p95"]], axis=1),
                    hovertemplate="%{x|%Y}: mediana %{y:,.0f} €<br>90 %: %{customdata[0]:,.0f} – %{customdata[1]:,.0f} €<extra></extra>")
    fig.add_scatter(x=historial["fecha"], y=historial["valor_mercado_eur"], mode="lines+markers",
                    line=dict(color=estilo.COLOR_PRIMARIO, width=2.5), marker=dict(size=5), name="Valor histórico",
                    customdata=historial["equipo"], hovertemplate="%{x|%d/%m/%Y}: %{y:,.0f} €<br>%{customdata}<extra></extra>")
    fig.add_vline(x=fecha0, line_dash="dot", line_color="#94A3B8")
    fig.update_layout(height=500, hovermode="x unified", legend=dict(orientation="h", y=-0.15),
                      yaxis_title="Valor de mercado (€)", xaxis_title=None, margin=dict(t=30, l=10, r=10, b=10))
    if log:
        # Marcas 1-2-5 en euros legibles (por defecto Plotly rotula "2, 3, 4"
        # dentro de cada década, sin unidades).
        minimo = max(min(historial["valor_mercado_eur"].min(), pct["p5"].min()), m.VALOR_MINIMO)
        maximo = max(historial["valor_mercado_eur"].max(), pct["p95"].max())
        marcas = [b * 10 ** e for e in range(3, 11) for b in (1, 2, 5) if minimo / 1.5 <= b * 10 ** e <= maximo * 1.5]
        fig.update_yaxes(type="log", tickvals=marcas, ticktext=[estilo.euros(v) for v in marcas])
    return fig


def _pestana_distribucion(caminos: np.ndarray, resumen: dict, anios: int) -> None:
    final = caminos[:, -1]
    fig = px.histogram(x=np.log10(final), nbins=60, labels={"x": "Valor final (€, escala log)"})
    fig.update_traces(marker_color=estilo.COLOR_ACENTO, opacity=0.8, hovertemplate="%{y} simulaciones<extra></extra>")
    marcas = [(resumen["v0"], "Hoy", "#0F172A"), (resumen["mediana"], "Mediana", estilo.COLOR_PRIMARIO),
              (resumen["p5"], "P5", "#EF4444"), (resumen["p95"], "P95", "#10B981")]
    for valor, texto, color in marcas:
        fig.add_vline(x=np.log10(valor), line_color=color, line_dash="dash",
                      annotation_text=f"{texto}: {estilo.euros(valor)}", annotation_font_color=color)
    ticks = [v for v in [1e4, 1e5, 1e6, 1e7, 1e8, 1e9] if np.log10(final.min()) - 0.5 <= np.log10(v) <= np.log10(final.max()) + 0.5]
    fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[estilo.euros(v) for v in ticks])
    fig.update_layout(height=360, yaxis_title="Nº de simulaciones", bargap=0.02, margin=dict(t=40, l=10, r=10, b=10),
                      title=f"Distribución del valor dentro de {anios} año(s)")
    c_graf, c_riesgo = st.columns([2, 1], gap="large")
    with c_graf:
        st.plotly_chart(fig, use_container_width=True)
    with c_riesgo:
        st.markdown("**Métricas de riesgo**")
        st.dataframe(pd.DataFrame([
            ("Valor esperado (media)", estilo.euros(resumen["media"])),
            ("Probabilidad de duplicar su valor", f"{resumen['prob_x2']:.0%}"),
            ("Probabilidad de perder la mitad", f"{resumen['prob_mitad']:.0%}"),
            ("VaR 95 %", estilo.euros(resumen["var95"])),
            ("CVaR 95 %", estilo.euros(resumen["cvar95"])),
        ], columns=["Métrica", "Valor"]), hide_index=True, use_container_width=True)
        st.caption("**VaR 95 %**: pérdida que solo se supera en el 5 % peor de los escenarios. **CVaR 95 %**: "
                   "pérdida media dentro de ese 5 % peor. Si son mayores que el valor actual es porque en esos "
                   "escenarios el valor no baja: se muestra la pérdida como negativa.")


def _pestana_anio_a_anio(caminos: np.ndarray, fecha0: pd.Timestamp) -> None:
    pct = m.percentiles_por_anio(caminos)
    pct["fecha"] = [f"{(fecha0 + pd.DateOffset(years=int(a))):%Y}" for a in pct["anio"]]
    pct["prob_sube"] = (caminos > caminos[0, 0]).mean(axis=0)
    columnas = {"fecha": "Año", "p5": "P5 (pesimista)", "p25": "P25", "p50": "Mediana", "p75": "P75",
                "p95": "P95 (optimista)", "media": "Media", "prob_sube": "Prob. de estar por encima de hoy"}
    tabla = pct[list(columnas)].iloc[1:].rename(columns=columnas)
    config = {c: st.column_config.NumberColumn(c, format="euro") for c in list(columnas.values())[1:7]}
    config["Prob. de estar por encima de hoy"] = st.column_config.ProgressColumn(
        "Prob. de estar por encima de hoy", format="percent", min_value=0.0, max_value=1.0)
    st.dataframe(tabla, hide_index=True, use_container_width=True, column_config=config)


def _pestana_volatilidad(ret_jugador: pd.DataFrame, retornos: pd.DataFrame, params: m.ParametrosCarrera,
                         edad0: int, etapa0: int, tramo0: int, posicion: str | None) -> None:
    mu_j, sigma_j, n_j = m.volatilidad_jugador(ret_jugador)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if ret_jugador.empty:
            st.info("El jugador no tiene dos años consecutivos con valoración: no se puede calcular su volatilidad.")
        else:
            datos = ret_jugador.assign(variacion=np.exp(ret_jugador["log_ret"]) - 1)
            fig = px.bar(datos, x="anio", y="variacion", labels={"anio": "Año", "variacion": "Variación anual"},
                         title="Variación anual de su valor")
            fig.update_traces(marker_color=np.where(datos["variacion"] >= 0, "#10B981", "#EF4444"),
                              hovertemplate="%{x}: %{y:+.0%}<extra></extra>")
            fig.update_yaxes(tickformat="+.0%")
            fig.update_layout(height=340, margin=dict(t=50, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        a, e, t = (int(i[0]) for i in params.indices([edad0], [etapa0], [tramo0]))
        estilo.tarjetas_kpi([
            (f"Volatilidad del jugador (últ. {n_j} años)",
             f"{sigma_j:.0%}" if not np.isnan(sigma_j) else "—", f"Tendencia: {mu_j:+.0%} anual (log)" if not np.isnan(mu_j) else ""),
            ("Volatilidad de su grupo", f"{params.sigma[a, e, t]:.0%}",
             f"Tendencia: {params.mu[a, e, t]:+.0%} · {params.n[a, e, t]:,} casos".replace(",", ".")),
        ])
        st.caption(f"Grupo usado para este jugador ({edad0} años, año de carrera {etapa0 + 1}): "
                   f"**{params.nivel[a, e, t]}**.")
    with c2:
        datos = retornos[retornos["tiene_edad"] & (retornos["tramo"] == tramo0)]
        if posicion is not None:
            datos = datos[datos["posicion"] == posicion]
        ciclo = (datos.assign(edad=datos["edad"].astype(float).clip(m.EDAD_MIN, m.EDAD_MAX).astype(int))
                 .groupby("edad")["log_ret"].agg(["mean", "std", "count"]).reset_index())
        ciclo = ciclo[ciclo["count"] >= 20]
        fig = go.Figure()
        fig.add_bar(x=ciclo["edad"], y=ciclo["mean"], name="Tendencia (μ)", marker_color=estilo.COLOR_PRIMARIO,
                    customdata=ciclo["count"], hovertemplate="%{y:+.0%} (%{customdata} casos)")
        fig.add_scatter(x=ciclo["edad"], y=ciclo["std"], name="Volatilidad (σ)", mode="lines+markers",
                        line=dict(color="#F59E0B", width=2.5), hovertemplate="%{y:.0%}")
        fig.add_vline(x=min(max(edad0, m.EDAD_MIN), m.EDAD_MAX), line_dash="dot", line_color="#0F172A",
                      annotation_text="Hoy")
        fig.update_layout(height=400, title=f"Tendencia y volatilidad anual por edad · tramo {m.NOMBRES_TRAMO[tramo0]}",
                          xaxis_title="Edad (años)", yaxis_tickformat="+.0%", legend=dict(orientation="h", y=-0.2),
                          margin=dict(t=50, l=10, r=10, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Estimadas con los jugadores del mismo nivel de valor"
                   f"{f' y posición ({posicion})' if posicion else ''}. "
                   "Así se ve el ciclo de vida típico: el valor sube de joven, se estanca hacia los 26–28 años "
                   "y cae a partir de los 30.")


def _pestana_validacion(retornos: pd.DataFrame) -> None:
    tabla = _validacion(retornos, len(retornos))
    st.markdown(
        f"Backtest a 1 año **fuera de muestra**: parámetros estimados solo con las variaciones que empiezan hasta "
        f"{tabla.attrs['anio_corte']} ({estilo.numero(tabla.attrs['n_entrenamiento'])} casos) y evaluados con las "
        f"posteriores ({estilo.numero(tabla.attrs['n_prueba'])} casos)."
    )
    st.dataframe(
        tabla, hide_index=True, use_container_width=True,
        column_config={
            "modelo": "Modelo",
            "cobertura_90": st.column_config.NumberColumn("Cobertura del intervalo 90 %", format="percent"),
            "cobertura_50": st.column_config.NumberColumn("Cobertura del intervalo 50 %", format="percent"),
            "acierto_direccion": st.column_config.NumberColumn("Acierto sube/baja", format="percent"),
            "error_medio": st.column_config.NumberColumn("Error medio (log)", format="%.4f"),
            "nll": st.column_config.NumberColumn("Log-verosimilitud neg. (menor = mejor)", format="%.4f"),
        },
    )
    st.caption("Un modelo bien calibrado contiene ~90 % de los valores reales en su intervalo del 90 % y ~50 % en el "
               "del 50 %. Cada caso se evalúa con los parámetros de su posición. Las versiones «sin edad» son "
               "los mismos modelos usando solo posición, año de carrera y tramo: "
               "añadir la edad reduce el error medio y la log-verosimilitud negativa (el modelo describe mejor "
               "la distribución real). El GBM con parámetros globales (sin ciclo de vida) es claramente peor.")


def _pestana_metodologia() -> None:
    st.markdown("#### Modelo común: Movimiento Browniano Geométrico")
    st.markdown("Igual que el precio de una acción, el valor se simula en escala logarítmica, año a año:")
    st.latex(r"\ln V_{t+1} = \ln V_t + \mu + \sigma Z_t, \qquad Z_t \sim \mathcal{N}(0,1)")
    st.markdown(
        "- **μ (tendencia)**: variación logarítmica media anual.\n"
        "- **σ (volatilidad)**: desviación típica anual de esa variación.\n"
        "- **Monte Carlo**: se repite miles de veces y el abanico de trayectorias se resume con percentiles "
        "(la mediana es el escenario central; P5 y P95, el pesimista y el optimista)."
    )
    st.markdown("#### Los dos modelos")
    for nombre, texto in MODELOS.items():
        st.markdown(f"- **{nombre}**: {texto}")
    st.markdown("#### Supuestos y limitaciones")
    st.markdown(
        "- μ y σ se estiman con los jugadores de la misma **posición** y dependen de la **edad** (calculada con la "
        "fecha de nacimiento), del **año de carrera** (años desde la primera valoración) y del **tramo de valor**. "
        "Cada año simulado el jugador cumple un año más, así que sus parámetros cambian con él.\n"
        f"- Si un grupo tiene menos de {m.MIN_OBSERVACIONES} casos se usa uno más amplio, en este orden: posición + "
        "edad + año de carrera + tramo → posición + edad + tramo → posición + edades cercanas (±1 a ±4 años) + tramo → "
        "edad + año de carrera + tramo → edad + tramo → edades cercanas + tramo → tramo → todos los jugadores.\n"
        "- Los jugadores sin posición conocida se comparan por **edad + tramo** con todos los jugadores.\n"
        f"- Las edades por debajo de {m.EDAD_MIN} y por encima de {m.EDAD_MAX} años se agrupan con esos extremos.\n"
        f"- Se usan solo los **últimos {m.VENTANA_ANIOS} años** de datos: con todo el histórico (desde 2004) el "
        "modelo sobrestimaba la subida de los jugadores de 19 a 27 años, porque el mercado crecía más entonces.\n"
        "- Los parámetros se estiman con la serie anual (último valor de cada año) y solo con años consecutivos.\n"
        "- No se modelan lesiones, cambios de club ni la retirada; el valor nunca baja de 10.000 € (mínimo de "
        "Transfermarkt).\n"
        "- Es una simulación estadística del comportamiento **típico** de jugadores en la misma situación, no una "
        "predicción individual: úsala como rango de escenarios, no como valor exacto."
    )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
def render() -> None:
    df = cargar_tabla()
    perfiles = perfil_jugadores(df, float(len(df)))
    anual, retornos = _datos(df, len(df))

    estilo.cabecera(
        "Predicción de valores de mercado",
        "Simula miles de posibles carreras de un jugador para ver hacia dónde puede ir su valor en los próximos años.",
    )

    ids = perfiles.sort_values("valor_actual", ascending=False).index.tolist()
    por_defecto = int(perfiles.index[perfiles["nombre"] == "Lamine Yamal"][0]) if (perfiles["nombre"] == "Lamine Yamal").any() else ids[0]
    c_jug, c_horiz, c_sim = st.columns([2.6, 1.2, 1.2], gap="large")
    id_player = c_jug.selectbox("Jugador", ids, index=ids.index(por_defecto), format_func=etiquetas_jugadores(perfiles, float(len(perfiles)), True).get,
                                key=CLAVE + "jugador", help="Ordenados por valor actual. Escribe para buscar.")
    anios = c_horiz.slider("Horizonte (años)", 1, 10, 5, key=CLAVE + "anios")
    n_sim = c_sim.select_slider("Nº de simulaciones", [1_000, 2_000, 5_000, 10_000, 20_000], value=5_000,
                                format_func=estilo.numero, key=CLAVE + "n")

    c_mod, _ = st.columns([2.6, 2.4], gap="large")
    modelo = c_mod.segmented_control("Modelo", list(MODELOS), default=MODELO_MC, key=CLAVE + "modelo") or MODELO_MC
    c_mod.caption(MODELOS[modelo])

    p = perfiles.loc[id_player]
    serie = anual[anual["id_player"] == id_player]
    historial = df[df["id_player"] == id_player].sort_values("fecha")
    ret_jugador = retornos[retornos["id_player"] == id_player].sort_values("anio")
    v0, fecha0 = float(historial["valor_mercado_eur"].iloc[-1]), historial["fecha"].iloc[-1]
    etapa0 = int(serie["etapa"].iloc[-1])
    tramo0 = int(m.tramo_de([v0])[0])
    if "edad" not in historial.columns or pd.isna(historial["edad"].iloc[-1]):
        st.warning("La tabla de datos no incluye la edad de este jugador, necesaria para la simulación. Copia en "
                   "`datos/` una versión de la tabla limpia que incluya la fecha de nacimiento.")
        return
    edad0 = int(historial["edad"].iloc[-1])
    # Grupo de comparación: jugadores de su posición; sin posición, solo edad + tramo.
    posicion = str(p["posicion"]) if pd.notna(p["posicion"]) else None
    params = _parametros(retornos, len(retornos), posicion)

    if (df["fecha"].max() - fecha0).days > 540:
        st.warning(f"La última valoración de este jugador es de {fecha0:%m/%Y}: puede estar retirado o sin club. "
                   "La simulación parte de ese último valor conocido.")

    caminos = _simular(modelo, v0, edad0, etapa0, anios, n_sim, params, f"{posicion}")
    resumen = m.resumen_final(caminos)
    cambio = resumen["mediana"] / v0 - 1

    st.divider()
    estilo.tarjetas_kpi([
        ("Valor actual", estilo.euros(v0), f"{fecha0:%d/%m/%Y}"),
        (f"Mediana a {anios} año(s)", estilo.euros(resumen["mediana"]), f"{cambio:+.0%} respecto a hoy"),
        ("Probabilidad de que suba", f"{resumen['prob_sube']:.0%}", f"Duplicar su valor: {resumen['prob_x2']:.0%}"),
        ("Rango del 90 %", f"{estilo.euros(resumen['p5'])} – {estilo.euros(resumen['p95'])}",
         "Escenario pesimista (P5) – optimista (P95)"),
    ])
    st.write("")

    c_ficha, c_grafico = st.columns([1, 3.2], gap="large")
    with c_ficha:
        _ficha(p, edad0, etapa0)
        st.write("")
        tendencia = ("🟢 **al alza**" if cambio > 0.1 else "🔴 **a la baja**" if cambio < -0.1 else "🟡 **estable**")
        st.markdown(
            f"En el escenario central, su valor evoluciona {tendencia}: de {estilo.euros(v0)} a unos "
            f"**{estilo.euros(resumen['mediana'])}** dentro de {anios} año(s). En el 90 % de las "
            f"{estilo.numero(n_sim)} simulaciones acaba entre {estilo.euros(resumen['p5'])} y "
            f"{estilo.euros(resumen['p95'])}."
        )
        log = st.toggle("Escala logarítmica", value=True, key=CLAVE + "log")
        ver_caminos = st.toggle("Ver 30 trayectorias simuladas", value=False, key=CLAVE + "caminos")
    with c_grafico:
        st.plotly_chart(_grafico_abanico(historial, caminos, fecha0, log, ver_caminos), use_container_width=True)

    tabs = st.tabs(["📊 Distribución final", "📅 Año a año", "📉 Volatilidad", "✅ Validación", "ℹ️ Cómo funciona"])
    with tabs[0]:
        _pestana_distribucion(caminos, resumen, anios)
    with tabs[1]:
        _pestana_anio_a_anio(caminos, fecha0)
    with tabs[2]:
        _pestana_volatilidad(ret_jugador, retornos, params, edad0, etapa0, tramo0, posicion)
    with tabs[3]:
        _pestana_validacion(retornos)
    with tabs[4]:
        _pestana_metodologia()
