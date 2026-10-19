"""
Modelos de simulación del valor de mercado de un jugador, inspirados en los
que se usan en finanzas para simular el precio de un activo.

Punto de partida: la serie ANUAL de cada jugador (último valor de cada año
natural) y sus rentabilidades logarítmicas anuales entre años consecutivos,
r = ln(V_{t+1} / V_t). Igual que con una acción, se simula ln(V) año a año:

    ln V_{t+1} = ln V_t + mu + sigma * Z,   Z ~ N(0, 1)

es decir, un Movimiento Browniano Geométrico (GBM) discreto, donde `mu` es la
tendencia (rentabilidad logarítmica media anual) y `sigma` la volatilidad
(desviación típica anual de esa rentabilidad). La simulación de Monte Carlo
repite el proceso miles de veces y resume el abanico de trayectorias con
percentiles.

A diferencia de una acción, el valor de un futbolista tiene un "ciclo de
vida" muy marcado (sube al principio de la carrera y cae con la edad) y
depende de su nivel (un jugador de 100 M€ apenas tiene margen para subir).
Por eso `mu` y `sigma` dependen, en cada año simulado y para cada
trayectoria, de tres cosas:

  - la EDAD del jugador en ese año;
  - su AÑO DE CARRERA (años desde su primera valoración): distingue, a la
    misma edad, a quien lleva años en la élite de quien acaba de llegar;
  - el TRAMO DE VALOR en el que esté esa trayectoria en ese momento.

Se estiman con los jugadores de su misma POSICIÓN (los que no tienen posición,
con edad + tramo de todos los jugadores) en los últimos `VENTANA_ANIOS` años. Con edad + año de carrera + tramo se
explica un ~10 % más de la variación anual del valor que solo con año de
carrera + tramo.

Dos modelos:

  1. `simular_montecarlo` (GBM por edad y etapa de carrera): Z normal.
  2. `simular_bootstrap` (simulación histórica): en vez de suponer una normal
     se toma al azar una rentabilidad REAL de jugadores en la misma situación
     (edad, año de carrera y tramo). No asume normalidad: recoge las colas
     gruesas de los datos (curtosis ~4).

El valor simulado nunca baja de 10.000 € (el mínimo que asigna Transfermarkt).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VALOR_MINIMO = 10_000.0
MAX_ETAPA = 15                 # años de carrera 15 o más se tratan juntos
EDAD_MIN, EDAD_MAX = 16, 38    # edades fuera de este rango se tratan como el extremo
# Tramos de valor. Los de élite se separan porque su dinámica es distinta: p.
# ej. en el año de carrera 8, un jugador de 10–30 M€ pierde de media ~17 %
# (log) al año y uno de más de 70 M€ solo ~6 %.
BORDES_TRAMO = [0, 250_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000, 70_000_000, np.inf]
NOMBRES_TRAMO = ["< 250 k€", "250 k€ – 1 M€", "1 – 3 M€", "3 – 10 M€", "10 – 30 M€", "30 – 70 M€", "> 70 M€"]
MIN_OBSERVACIONES = 40         # por debajo, se usan parámetros de un grupo más amplio
# Edades vecinas que se van uniendo cuando un grupo no tiene bastantes casos.
# Hasta ±4 para que los jugadores jóvenes de élite (p. ej. 18 años y > 70 M€)
# no pierdan la edad y la posición y caigan al nivel "solo tramo".
RADIOS_EDAD = (1, 2, 3, 4)
# Años (hasta el último con datos) con los que se estiman los parámetros. Con
# todo el histórico (desde 2004) el modelo sobrestimaba ~5-10 puntos al año la
# subida de los jugadores de 19 a 27 años; con 8 años el sesgo se reduce a la
# mitad sin perder precisión (con 5, pesa demasiado la caída de 2020).
VENTANA_ANIOS = 8
PERCENTILES = [5, 25, 50, 75, 95]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def serie_anual(df: pd.DataFrame) -> pd.DataFrame:
    """Último valor de cada jugador en cada año natural, con su edad en ese
    registro y su año de carrera (`etapa`: años desde su primer año con
    valoración)."""
    agregaciones = dict(valor=("valor_mercado_eur", "last"), posicion=("posicion", "last"), fecha=("fecha", "last"))
    if "edad" in df.columns:
        agregaciones["edad"] = ("edad", "last")
    if "equipo" in df.columns:
        agregaciones["equipo"] = ("equipo", "last")
    anual = (
        df.assign(anio=df["fecha"].dt.year)
        .sort_values("fecha")
        .groupby(["id_player", "anio"], sort=False)
        .agg(**agregaciones)
        .reset_index()
        .sort_values(["id_player", "anio"])
    )
    if "edad" not in anual.columns:
        anual["edad"] = pd.NA
    anual["etapa"] = anual["anio"] - anual.groupby("id_player")["anio"].transform("min")
    return anual.reset_index(drop=True)


def tramo_de(valores) -> np.ndarray:
    """Índice de tramo de valor (0..6) de cada valor."""
    return np.clip(np.searchsorted(BORDES_TRAMO, np.asarray(valores, dtype=float), side="right") - 1,
                   0, len(NOMBRES_TRAMO) - 1)


def indice_edad(edades) -> np.ndarray:
    """Posición de cada edad en los ejes de parámetros (EDAD_MIN..EDAD_MAX)."""
    return np.clip(np.asarray(edades, dtype=float), EDAD_MIN, EDAD_MAX).astype(int) - EDAD_MIN


def retornos_anuales(anual: pd.DataFrame) -> pd.DataFrame:
    """Rentabilidad logarítmica entre dos años CONSECUTIVOS del mismo jugador
    (los años sin valoración rompen la serie: no se interpolan). La edad, la
    etapa y el tramo son los del año de partida."""
    siguiente = anual.groupby("id_player")[["anio", "valor"]].shift(-1)
    consecutivo = siguiente["anio"] == anual["anio"] + 1
    r = anual[consecutivo].copy()
    r["valor_fin"] = siguiente.loc[consecutivo, "valor"].to_numpy()
    r["log_ret"] = np.log(r["valor_fin"] / r["valor"])
    r["etapa_c"] = r["etapa"].clip(upper=MAX_ETAPA)
    r["tramo"] = tramo_de(r["valor"])
    r["edad_i"] = indice_edad(r["edad"].fillna(-1)) if r["edad"].notna().any() else 0
    r["tiene_edad"] = r["edad"].notna()
    return r.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parámetros por edad, etapa y tramo (con respaldo jerárquico)
# ---------------------------------------------------------------------------
def _grupos(datos: pd.DataFrame, claves: list[str]) -> dict:
    """Rentabilidades de cada grupo. Con una sola clave, pandas devuelve las
    claves como tuplas de un elemento `(6,)`: se desempaquetan a `6`."""
    grupos = datos.groupby(claves if len(claves) > 1 else claves[0])["log_ret"]
    return {k: v.to_numpy() for k, v in grupos}


def _vecinas(grupos: dict, x: int, t: int, radio: int) -> np.ndarray | None:
    """Une los grupos (x ± radio, t) de un diccionario indexado por (x, tramo):
    edades (o años de carrera) cercanas dentro del mismo tramo."""
    partes = [grupos[(v, t)] for v in range(x - radio, x + radio + 1) if (v, t) in grupos]
    return np.concatenate(partes) if partes else None


class ParametrosCarrera:
    """mu, sigma y rentabilidades observadas para cada (edad, año de carrera,
    tramo de valor), calculadas con los jugadores de la misma POSICIÓN. Si un
    grupo tiene menos de `MIN_OBSERVACIONES`, se usa el de un nivel más
    general, por este orden:

        posición + edad + etapa + tramo
        posición + edad + tramo
        posición + edades cercanas (±1 … ±4) + tramo
        edad + etapa + tramo
        edad + tramo
        edades cercanas (±1 … ±4) + tramo
        tramo
        global

    Para jugadores sin posición (`posicion=None`) el grupo es edad + tramo
    (y después edades cercanas, tramo y global).

    Con `usar_edad=False` se ignora la edad (posición + etapa + tramo, etapa +
    tramo...), lo que permite medir en la validación cuánto aporta la edad.

    Solo se usan las rentabilidades de los últimos `VENTANA_ANIOS` años de
    `retornos` (en la validación, los anteriores al corte)."""

    def __init__(self, retornos: pd.DataFrame, posicion: str | None = None, usar_edad: bool = True,
                 ventana_anios: int | None = VENTANA_ANIOS):
        self.usar_edad = usar_edad
        self.posicion = posicion
        if ventana_anios and len(retornos):
            # Solo los últimos años: el mercado de hace 15-20 años crecía más
            # y, con todo el histórico, el modelo era optimista con los
            # jugadores de 19 a 27 años (validado fuera de muestra).
            retornos = retornos[retornos["anio"] > retornos["anio"].max() - ventana_anios]
        n_edades = (EDAD_MAX - EDAD_MIN + 1) if usar_edad else 1
        forma = (n_edades, MAX_ETAPA + 1, len(NOMBRES_TRAMO))
        self.mu, self.sigma = np.zeros(forma), np.zeros(forma)
        self.n = np.zeros(forma, dtype=int)
        self.nivel = np.empty(forma, dtype=object)
        self.muestras: dict[tuple[int, int, int], np.ndarray] = {}

        datos = retornos[retornos["tiene_edad"]] if usar_edad else retornos
        propia = datos[datos["posicion"] == posicion] if posicion is not None else None
        todos = datos["log_ret"].to_numpy()
        # Cada nivel: (nombre, función (a, e, t) -> rentabilidades del grupo o None).
        niveles: list[tuple[str, callable]] = []
        if usar_edad:
            if propia is not None:
                p_aet = _grupos(propia, ["edad_i", "etapa_c", "tramo"])
                p_at = _grupos(propia, ["edad_i", "tramo"])
                aet = _grupos(datos, ["edad_i", "etapa_c", "tramo"])
                niveles += [
                    ("posición + edad + etapa + tramo", lambda a, e, t: p_aet.get((a, e, t))),
                    ("posición + edad + tramo", lambda a, e, t: p_at.get((a, t))),
                ]
                niveles += [(f"posición + edad ±{r} + tramo", lambda a, e, t, r=r: _vecinas(p_at, a, t, r))
                            for r in RADIOS_EDAD]
                niveles.append(("edad + etapa + tramo", lambda a, e, t: aet.get((a, e, t))))
            at = _grupos(datos, ["edad_i", "tramo"])
            niveles.append(("edad + tramo", lambda a, e, t: at.get((a, t))))
            niveles += [(f"edad ±{r} + tramo", lambda a, e, t, r=r: _vecinas(at, a, t, r)) for r in RADIOS_EDAD]
        else:
            if propia is not None:
                p_et = _grupos(propia, ["etapa_c", "tramo"])
                niveles.append(("posición + etapa + tramo", lambda a, e, t: p_et.get((e, t))))
            et = _grupos(datos, ["etapa_c", "tramo"])
            niveles += [("etapa + tramo", lambda a, e, t: et.get((e, t)))]
            niveles += [(f"etapa ±{r} + tramo", lambda a, e, t, r=r: _vecinas(et, e, t, r)) for r in (1, 2, 3)]
        por_tramo = _grupos(datos, ["tramo"])
        niveles.append(("tramo", lambda a, e, t: por_tramo.get(t)))

        for a in range(forma[0]):
            for e in range(forma[1]):
                for t in range(forma[2]):
                    muestra, nivel = todos, "global"
                    for nombre, grupo in niveles:
                        candidata = grupo(a, e, t)
                        if candidata is not None and len(candidata) >= MIN_OBSERVACIONES:
                            muestra, nivel = candidata, nombre
                            break
                    self.mu[a, e, t] = muestra.mean()
                    self.sigma[a, e, t] = muestra.std(ddof=1)
                    self.n[a, e, t] = len(muestra)
                    self.nivel[a, e, t] = nivel
                    self.muestras[(a, e, t)] = muestra

    def indices(self, edades, etapas, tramos):
        """Índices (edad, etapa, tramo) sobre los ejes de parámetros."""
        a = indice_edad(edades) if self.usar_edad else np.zeros(np.shape(tramos), dtype=int)
        return a, np.minimum(np.asarray(etapas, dtype=int), MAX_ETAPA), np.asarray(tramos, dtype=int)


# ---------------------------------------------------------------------------
# Simulación de Monte Carlo
# ---------------------------------------------------------------------------
def simular_montecarlo(v0: float, edad0: int, etapa0: int, anios: int, params: ParametrosCarrera, n: int,
                       semilla: int = 0) -> np.ndarray:
    """GBM cuyos mu/sigma dependen, cada año simulado, de la edad y el año de
    carrera de ese año y del tramo de valor ACTUAL de cada trayectoria."""
    rng = np.random.default_rng(semilla)
    log_v = np.full(n, np.log(v0))
    caminos = [np.full(n, v0)]
    for y in range(anios):
        a, e, t = params.indices(np.full(n, edad0 + y), np.full(n, etapa0 + y), tramo_de(np.exp(log_v)))
        log_v = log_v + params.mu[a, e, t] + params.sigma[a, e, t] * rng.standard_normal(n)
        log_v = np.maximum(log_v, np.log(VALOR_MINIMO))
        caminos.append(np.exp(log_v))
    return np.column_stack(caminos)


def simular_bootstrap(v0: float, edad0: int, etapa0: int, anios: int, params: ParametrosCarrera, n: int,
                      semilla: int = 0) -> np.ndarray:
    """Simulación histórica: cada año, cada trayectoria toma al azar una
    rentabilidad REAL observada en su misma edad, año de carrera y tramo."""
    rng = np.random.default_rng(semilla)
    log_v = np.full(n, np.log(v0))
    caminos = [np.full(n, v0)]
    for y in range(anios):
        a, e, t = params.indices(np.full(n, edad0 + y), np.full(n, etapa0 + y), tramo_de(np.exp(log_v)))
        incremento = np.empty(n)
        for tramo in np.unique(t):
            mascara = t == tramo
            clave = (int(a[0]), int(e[0]), int(tramo))
            incremento[mascara] = rng.choice(params.muestras[clave], size=int(mascara.sum()))
        log_v = np.maximum(log_v + incremento, np.log(VALOR_MINIMO))
        caminos.append(np.exp(log_v))
    return np.column_stack(caminos)


def volatilidad_jugador(retornos_jugador: pd.DataFrame, ventana: int = 5) -> tuple[float, float, int]:
    """Tendencia y volatilidad del propio jugador en sus `ventana` últimas
    rentabilidades anuales (solo informativo: se compara con la de su grupo)."""
    r = retornos_jugador.sort_values("anio")["log_ret"].to_numpy()[-ventana:]
    if len(r) < 2:
        return float("nan"), float("nan"), len(r)
    return float(r.mean()), float(r.std(ddof=1)), len(r)


# ---------------------------------------------------------------------------
# Resumen de la simulación
# ---------------------------------------------------------------------------
def percentiles_por_anio(caminos: np.ndarray) -> pd.DataFrame:
    tabla = pd.DataFrame(np.percentile(caminos, PERCENTILES, axis=0).T, columns=[f"p{p}" for p in PERCENTILES])
    tabla["media"] = caminos.mean(axis=0)
    tabla["anio"] = range(caminos.shape[1])
    return tabla


def resumen_final(caminos: np.ndarray) -> dict:
    """Métricas del valor al final del horizonte, con el lenguaje de riesgo
    de finanzas: probabilidad de subir, múltiplos, VaR y CVaR al 95 %."""
    v0, final = caminos[0, 0], caminos[:, -1]
    p5 = np.percentile(final, 5)
    return {
        "v0": v0,
        "mediana": float(np.median(final)),
        "media": float(final.mean()),
        "p5": float(p5),
        "p95": float(np.percentile(final, 95)),
        "prob_sube": float((final > v0).mean()),
        "prob_x2": float((final >= 2 * v0).mean()),
        "prob_mitad": float((final <= 0.5 * v0).mean()),
        "var95": float(v0 - p5),
        "cvar95": float(v0 - final[final <= p5].mean()),
    }


# ---------------------------------------------------------------------------
# Validación fuera de muestra
# ---------------------------------------------------------------------------
def _metricas(nombre: str, r: np.ndarray, centro: np.ndarray, bajo90, alto90, bajo50, alto50,
              sigma: np.ndarray | None = None) -> dict:
    """Cobertura de los intervalos, acierto de dirección, error medio y, para
    los modelos normales, log-verosimilitud negativa media (NLL: cuanto MENOR,
    mejor describe el modelo la distribución real; es la métrica más sensible
    a si mu Y sigma son los adecuados para cada caso)."""
    con_cambio = r != 0
    nll = (float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * ((r - centro) / sigma) ** 2))
           if sigma is not None else float("nan"))
    return {
        "modelo": nombre,
        "cobertura_90": float(((r >= bajo90) & (r <= alto90)).mean()),
        "cobertura_50": float(((r >= bajo50) & (r <= alto50)).mean()),
        "acierto_direccion": float((np.sign(centro[con_cambio]) == np.sign(r[con_cambio])).mean()),
        "error_medio": float(np.mean(np.abs(r - centro))),
        "nll": nll,
    }


def validar(retornos: pd.DataFrame, anio_corte: int = 2018) -> pd.DataFrame:
    """Backtest a 1 año: parámetros estimados SOLO con rentabilidades que
    empiezan hasta `anio_corte`, evaluados con las posteriores. Compara los
    dos modelos CON edad frente a sus versiones SIN edad (solo etapa + tramo)
    y frente a un GBM con parámetros globales. Si un modelo está bien
    calibrado, su intervalo del 90 % contiene ~90 % de los valores reales y el
    del 50 %, ~50 %."""
    datos = retornos[retornos["tiene_edad"]]
    entrenamiento, prueba = datos[datos["anio"] <= anio_corte], datos[datos["anio"] > anio_corte]
    prueba = prueba.reset_index(drop=True)
    r = prueba["log_ret"].to_numpy()
    posiciones = prueba["posicion"].astype(object).where(prueba["posicion"].notna(), None)
    filas = []
    for usar_edad, etiqueta in ((True, "con edad"), (False, "sin edad")):
        mu, sigma, q = np.zeros(len(r)), np.zeros(len(r)), np.zeros((len(r), 5))
        # Cada caso se evalúa con los parámetros de su posición (o sin
        # posición si no la tiene), igual que en la simulación.
        for posicion, filas_pos in prueba.groupby(posiciones, dropna=False).groups.items():
            params = ParametrosCarrera(entrenamiento, posicion if pd.notna(posicion) else None, usar_edad)
            sub = prueba.loc[filas_pos]
            a, e, t = params.indices(sub["edad"].to_numpy(dtype=float), sub["etapa_c"], sub["tramo"])
            mu[filas_pos], sigma[filas_pos] = params.mu[a, e, t], params.sigma[a, e, t]
            # Bootstrap: cuantiles empíricos de cada grupo, calculados una vez por grupo.
            cuantiles = np.zeros(params.mu.shape + (5,))
            for clave, muestra in params.muestras.items():
                cuantiles[clave] = np.percentile(muestra, [5, 25, 50, 75, 95])
            q[filas_pos] = cuantiles[a, e, t]
        filas.append(_metricas(f"Monte Carlo ({etiqueta})", r, mu,
                               mu - 1.645 * sigma, mu + 1.645 * sigma, mu - 0.674 * sigma, mu + 0.674 * sigma, sigma))
        filas.append(_metricas(f"Simulación histórica ({etiqueta})", r, q[:, 2], q[:, 0], q[:, 4], q[:, 1], q[:, 3]))
    mu_g, sigma_g = entrenamiento["log_ret"].mean(), entrenamiento["log_ret"].std()
    centro = np.full(len(r), mu_g)
    filas.append(_metricas("GBM con parámetros globales (referencia)", r, centro, mu_g - 1.645 * sigma_g,
                           mu_g + 1.645 * sigma_g, mu_g - 0.674 * sigma_g, mu_g + 0.674 * sigma_g,
                           np.full(len(r), sigma_g)))
    tabla = pd.DataFrame(filas)
    tabla.attrs.update(n_entrenamiento=len(entrenamiento), n_prueba=len(prueba), anio_corte=anio_corte)
    return tabla
