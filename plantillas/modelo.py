"""
Gestión de plantillas: la plantilla de cada club estudiada como una cartera
de inversión, con la teoría de carteras de Markowitz.

1. ACTIVOS. Cada jugador es un activo y su peso en la cartera es la parte del
   valor de mercado de la plantilla que representa (w_i = V_i / suma de V).

2. SEGMENTOS. Los jugadores se agrupan en 16 segmentos: línea (portero,
   defensa, centrocampista, delantero) × grupo de edad (hasta 21, 22–25,
   26–29, 30 o más). La rentabilidad anual de un jugador es la de su segmento
   más una parte propia: R_i = F_s(i) + e_i.

3. MERCADO. Con el histórico de valoraciones de las cinco grandes ligas se
   construye, para cada segmento, un índice ponderado por valor (la variación
   anual del valor conjunto de sus jugadores). De esos índices salen la
   rentabilidad esperada de cada segmento (media de los últimos años), la
   covarianza entre segmentos (todos los años) y el riesgo propio de un
   jugador (varianza de e_i).

4. CARTERA. Para una plantilla con pesos w_i (W_s por segmento):

       rentabilidad esperada = suma_s W_s · mu_s
       varianza = W' Σ W  +  suma_i w_i² · d_s(i)
                  (riesgo de mercado)   (riesgo específico: concentración)

5. FRONTERA EFICIENTE. Para cada rentabilidad, el reparto por segmentos de
   menor riesgo, repartiendo el valor por igual entre `N_FRONTERA` jugadores
   y con cada línea dentro de los márgenes que se observan en los clubes
   reales (no hay plantillas sin porteros).

El valor de mercado es lo único que se mide: una plantilla "eficiente" lo es
como inversión, no necesariamente en lo deportivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

ARCHIVO = Path(__file__).resolve().parent.parent / "datos" / "valores_mercado_2026_R.csv"
LIGAS = ["England", "Spain", "Italy", "Germany", "France"]
LINEAS = ["Goalkeeper", "Defender", "Midfield", "Attack"]
LINEAS_ES = {"Goalkeeper": "Porteros", "Defender": "Defensas", "Midfield": "Centrocampistas", "Attack": "Delanteros"}
LINEA_DE = {
    "Goalkeeper": "Goalkeeper",
    "Centre-Back": "Defender", "Left-Back": "Defender", "Right-Back": "Defender",
    "Defensive Midfield": "Midfield", "Central Midfield": "Midfield", "Attacking Midfield": "Midfield",
    "Left Midfield": "Midfield", "Right Midfield": "Midfield",
    "Left Winger": "Attack", "Right Winger": "Attack", "Second Striker": "Attack", "Centre-Forward": "Attack",
}
CORTES_EDAD = [22, 26, 30]
EDADES = ["Hasta 21 años", "22–25 años", "26–29 años", "30 años o más"]
SEGMENTOS = [(linea, edad) for linea in LINEAS for edad in range(len(EDADES))]      # 16, en este orden
ANIO_MIN = 2005            # primer año del histórico usado (antes hay muy pocas valoraciones)
VENTANA_MU = 8             # años recientes con los que se estima la rentabilidad esperada (como en la predicción)
VALOR_MIN_PROPIO = 1_000_000   # el riesgo propio se mide con jugadores de al menos este valor
RECORTE_PROPIO = 3.0       # las variaciones propias se recortan a ±300 % (subidas extremas de jugadores baratos)
ENCOGIMIENTO = 0.5         # las correlaciones entre segmentos se acercan a la mitad a su media (pocos años)
N_FRONTERA = 25            # jugadores de la plantilla de referencia de la frontera
CUANTILES_LINEA = (0.05, 0.95)   # margen admitido para el peso de cada línea en la frontera


def grupo_edad(edades) -> np.ndarray:
    return np.searchsorted(CORTES_EDAD, np.asarray(edades, dtype=float), side="right")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_plantillas(ruta: Path | str = ARCHIVO) -> pd.DataFrame:
    """Plantillas de la temporada: un jugador por fila, con su línea, su grupo
    de edad, su segmento y su peso en el valor de la plantilla."""
    d = pd.read_csv(ruta).rename(columns={
        "country": "liga", "squad": "equipo", "player_name": "nombre", "player_position": "demarcacion",
        "player_age": "edad", "player_nationality": "nacionalidad", "player_foot": "pie",
        "player_market_value_euro": "valor", "season_start_year": "temporada"})
    d["altura_cm"] = d["player_height_mtrs"] * 100
    d["pie"] = d["pie"].str.strip().replace({"": pd.NA, "right": "Derecho", "left": "Izquierdo", "both": "Ambos"})
    d["posicion"] = d["demarcacion"].map(LINEA_DE)
    d = d[d["posicion"].notna() & (d["valor"] > 0)].reset_index(drop=True)
    d["g_edad"] = grupo_edad(d["edad"])
    d["segmento"] = d["posicion"].map(LINEAS.index) * len(EDADES) + d["g_edad"]
    d["peso"] = d["valor"] / d.groupby("equipo")["valor"].transform("sum")
    return d[["liga", "equipo", "nombre", "demarcacion", "posicion", "edad", "g_edad", "segmento", "nacionalidad",
              "altura_cm", "pie", "valor", "peso"]]


def retornos_historicos(df: pd.DataFrame) -> pd.DataFrame:
    """Variación anual simple (V_t+1 / V_t − 1) del valor de cada jugador de
    las cinco grandes ligas entre dos años consecutivos, a partir de la tabla
    limpia (último valor de cada año natural)."""
    anual = (df.assign(anio=df["fecha"].dt.year).sort_values("fecha")
             .groupby(["id_player", "anio"], sort=False)
             .agg(valor=("valor_mercado_eur", "last"), posicion=("posicion", "last"), edad=("edad", "last"),
                  liga=("liga", "last"))
             .reset_index().sort_values(["id_player", "anio"]))
    siguiente = anual.groupby("id_player")[["anio", "valor"]].shift(-1)
    seguido = siguiente["anio"] == anual["anio"] + 1
    r = anual[seguido].copy()
    r["ret"] = siguiente.loc[seguido, "valor"].to_numpy() / r["valor"].to_numpy() - 1
    r = r[r["liga"].isin(LIGAS) & r["posicion"].isin(LINEAS) & r["edad"].notna() & (r["anio"] >= ANIO_MIN)]
    r["segmento"] = r["posicion"].map(LINEAS.index).astype(int) * len(EDADES) + grupo_edad(r["edad"])
    return r.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mercado: rentabilidad, covarianza y riesgo propio de los segmentos
# ---------------------------------------------------------------------------
@dataclass
class Mercado:
    mu: np.ndarray              # (16,) rentabilidad anual esperada de cada segmento
    cov: np.ndarray             # (16, 16) covarianza entre los índices de los segmentos
    propio: np.ndarray          # (16,) varianza propia de un jugador del segmento
    indices: pd.DataFrame       # año × segmento: rentabilidad del índice ponderado por valor
    n: np.ndarray               # (16,) variaciones históricas de cada segmento
    correlacion_media: float

    @property
    def sigma(self) -> np.ndarray:
        """Volatilidad anual del índice de cada segmento."""
        return np.sqrt(np.diag(self.cov))


def estimar_mercado(retornos: pd.DataFrame, ventana_mu: int = VENTANA_MU, encogimiento: float = ENCOGIMIENTO) -> Mercado:
    r = retornos.assign(vr=retornos["valor"] * retornos["ret"])
    g = r.groupby(["anio", "segmento"])
    indices = (g["vr"].sum() / g["valor"].sum()).unstack("segmento").reindex(columns=range(len(SEGMENTOS)))
    mu = indices.tail(ventana_mu).mean().to_numpy()

    # Covarianza entre segmentos. Con una veintena de años y 16 segmentos las
    # correlaciones muestrales son muy ruidosas: se acercan a su media.
    sigma = indices.std().to_numpy()
    corr = indices.corr().to_numpy()
    fuera = ~np.eye(len(sigma), dtype=bool)
    media = float(corr[fuera].mean())
    corr = np.where(fuera, (1 - encogimiento) * corr + encogimiento * media, 1.0)
    cov = corr * np.outer(sigma, sigma)

    # Riesgo propio: lo que la variación de un jugador se aparta de la de su
    # segmento ese año, ponderado por valor (en la cartera pesan los caros).
    caros = r[r["valor"] >= VALOR_MIN_PROPIO]
    indice = pd.Series(indices.stack(), name="indice")
    e = (caros["ret"] - caros.join(indice, on=["anio", "segmento"])["indice"]).clip(-RECORTE_PROPIO, RECORTE_PROPIO)
    caros = caros.assign(ve2=caros["valor"] * e ** 2)
    por_segmento = caros.groupby("segmento")
    propio = (por_segmento["ve2"].sum() / por_segmento["valor"].sum()).reindex(range(len(SEGMENTOS))).to_numpy()
    return Mercado(mu=mu, cov=cov, propio=propio, indices=indices,
                   n=r.groupby("segmento").size().reindex(range(len(SEGMENTOS)), fill_value=0).to_numpy(),
                   correlacion_media=media)


# ---------------------------------------------------------------------------
# Carteras
# ---------------------------------------------------------------------------
def pesos_segmento(plantilla: pd.DataFrame) -> np.ndarray:
    """Parte del valor de la plantilla que hay en cada segmento."""
    return plantilla.groupby("segmento")["peso"].sum().reindex(range(len(SEGMENTOS)), fill_value=0.0).to_numpy()


def analizar(plantilla: pd.DataFrame, mercado: Mercado) -> dict:
    """Rentabilidad esperada y riesgo de una plantilla (los pesos de sus
    jugadores deben sumar 1)."""
    w = plantilla["peso"].to_numpy()
    W = pesos_segmento(plantilla)
    var_mercado = float(W @ mercado.cov @ W)
    var_propia = float(np.sum(w ** 2 * mercado.propio[plantilla["segmento"].to_numpy()]))
    hhi = float(np.sum(w ** 2))
    return {
        "jugadores": len(plantilla), "valor": float(plantilla["valor"].sum()), "rentabilidad": float(W @ mercado.mu),
        "riesgo": float(np.sqrt(var_mercado + var_propia)), "riesgo_mercado": float(np.sqrt(var_mercado)),
        "riesgo_propio": float(np.sqrt(var_propia)), "concentracion": hhi, "jugadores_efectivos": 1 / hhi,
        "mayor_peso": float(w.max()), "edad_media": float(np.sum(w * plantilla["edad"].to_numpy())),
    }


def analizar_clubes(plantillas: pd.DataFrame, mercado: Mercado) -> pd.DataFrame:
    """Una fila por club con su rentabilidad, su riesgo y su reparto por
    líneas y por grupos de edad."""
    filas = {}
    for equipo, plantilla in plantillas.groupby("equipo"):
        W = pesos_segmento(plantilla).reshape(len(LINEAS), len(EDADES))
        filas[equipo] = {"liga": plantilla["liga"].iloc[0], **analizar(plantilla, mercado),
                         **{f"linea_{l}": W[i].sum() for i, l in enumerate(LINEAS)},
                         **{f"edad_{j}": W[:, j].sum() for j in range(len(EDADES))},
                         **{f"seg_{k}": v for k, v in enumerate(W.ravel())}}
    return pd.DataFrame.from_dict(filas, orient="index").rename_axis("equipo")


def limites_linea(clubes: pd.DataFrame, cuantiles: tuple[float, float] = CUANTILES_LINEA) -> np.ndarray:
    """Peso mínimo y máximo de cada línea en las carteras de la frontera: el
    rango en el que se mueve la gran mayoría de los clubes."""
    return np.array([clubes[f"linea_{l}"].quantile(list(cuantiles)).to_numpy() for l in LINEAS])


def _restricciones(limites: np.ndarray):
    """Matriz y cotas de `A W <= b`: cada línea dentro de sus límites."""
    A = np.zeros((2 * len(LINEAS), len(SEGMENTOS)))
    b = np.zeros(2 * len(LINEAS))
    for i in range(len(LINEAS)):
        columnas = slice(i * len(EDADES), (i + 1) * len(EDADES))
        A[2 * i, columnas], b[2 * i] = 1, limites[i, 1]           # suma de la línea <= máximo
        A[2 * i + 1, columnas], b[2 * i + 1] = -1, -limites[i, 0]  # suma de la línea >= mínimo
    return A, b


def _varianza(W: np.ndarray, mercado: Mercado, n_jugadores: int) -> float:
    """Varianza de una cartera con pesos W por segmento cuyo valor se reparte
    por igual entre `n_jugadores` (el riesgo propio de un segmento con peso
    W_s repartido entre n·W_s jugadores es W_s · d_s / n)."""
    return float(W @ mercado.cov @ W + mercado.propio @ W / n_jugadores)


def _minimo_riesgo(mercado: Mercado, limites: np.ndarray, n_jugadores: int, rentabilidad: float | None = None,
                   inicio: np.ndarray | None = None) -> np.ndarray:
    """Pesos por segmento de mínima varianza (con una rentabilidad dada, si se
    indica)."""
    A, b = _restricciones(limites)
    k = len(SEGMENTOS)
    condiciones = [{"type": "eq", "fun": lambda W: W.sum() - 1, "jac": lambda W: np.ones(k)},
                   {"type": "ineq", "fun": lambda W: b - A @ W, "jac": lambda W: -A}]
    if rentabilidad is not None:
        condiciones.append({"type": "eq", "fun": lambda W: W @ mercado.mu - rentabilidad, "jac": lambda W: mercado.mu})
    if inicio is None:
        inicio = np.repeat(limites.mean(axis=1) / limites.mean(axis=1).sum() / len(EDADES), len(EDADES))
    res = minimize(lambda W: _varianza(W, mercado, n_jugadores), inicio,
                   jac=lambda W: 2 * mercado.cov @ W + mercado.propio / n_jugadores,
                   bounds=[(0, 1)] * k, constraints=condiciones, method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    return np.clip(res.x, 0, 1)


def frontera(mercado: Mercado, limites: np.ndarray, n_jugadores: int = N_FRONTERA, puntos: int = 60) -> pd.DataFrame:
    """Frontera eficiente: de la cartera de mínimo riesgo a la de máxima
    rentabilidad. Columnas: rentabilidad, riesgo y el peso de cada segmento."""
    A, b = _restricciones(limites)
    maxima = linprog(-mercado.mu, A_ub=A, b_ub=b, A_eq=np.ones((1, len(SEGMENTOS))), b_eq=[1], bounds=(0, 1))
    W = _minimo_riesgo(mercado, limites, n_jugadores)
    filas = []
    for objetivo in np.linspace(W @ mercado.mu, -maxima.fun, puntos):
        W = _minimo_riesgo(mercado, limites, n_jugadores, objetivo, inicio=W)
        filas.append([float(W @ mercado.mu), float(np.sqrt(_varianza(W, mercado, n_jugadores))), *W])
    return pd.DataFrame(filas, columns=["rentabilidad", "riesgo", *range(len(SEGMENTOS))])


def cartera_eficiente(curva: pd.DataFrame, riesgo: float) -> pd.Series:
    """Cartera de la frontera con el mismo riesgo (interpolando entre dos
    puntos de la curva). Si el riesgo es menor que el mínimo de la frontera,
    la de mínimo riesgo; si es mayor que el máximo, la de máxima rentabilidad."""
    x = curva["riesgo"].to_numpy()
    riesgo = float(np.clip(riesgo, x[0], x[-1]))
    return pd.Series({c: float(np.interp(riesgo, x, curva[c].to_numpy())) for c in curva.columns})


def riesgo_repartido(clubes: pd.DataFrame, mercado: Mercado, n_jugadores: int = N_FRONTERA) -> np.ndarray:
    """Riesgo que tendría cada club con su mismo reparto por segmentos pero
    con el valor repartido por igual entre `n_jugadores`, como en la frontera:
    separa lo que se debe a la concentración en pocos jugadores."""
    propio = np.array([sum(clubes[f"seg_{k}"] * mercado.propio[k] for k in range(len(SEGMENTOS)))]).ravel() / n_jugadores
    return np.sqrt(clubes["riesgo_mercado"].to_numpy() ** 2 + propio)


def comparar_con_frontera(clubes: pd.DataFrame, curva: pd.DataFrame, mercado: Mercado) -> pd.DataFrame:
    """Añade a cada club la rentabilidad de la cartera eficiente con su mismo
    riesgo, la brecha que le separa de ella, cuánto de esa brecha se debe al
    reparto por líneas y edades y cuánto a la concentración en pocos
    jugadores, y su puesto (1 = el más cercano a la frontera)."""
    eficiente = np.array([cartera_eficiente(curva, r)["rentabilidad"] for r in clubes["riesgo"]])
    repartido = riesgo_repartido(clubes, mercado)
    eficiente_repartido = np.array([cartera_eficiente(curva, r)["rentabilidad"] for r in repartido])
    out = clubes.assign(rentabilidad_eficiente=eficiente, brecha=eficiente - clubes["rentabilidad"],
                        riesgo_repartido=repartido, brecha_reparto=eficiente_repartido - clubes["rentabilidad"])
    out["brecha_concentracion"] = out["brecha"] - out["brecha_reparto"]
    out["puesto"] = out["brecha"].rank(method="min").astype(int)
    return out


def aportacion_al_riesgo(plantilla: pd.DataFrame, mercado: Mercado) -> np.ndarray:
    """Parte de la varianza de la plantilla que aporta cada jugador (suman 1):
    su peso por la covarianza de su segmento con el resto de la plantilla,
    más su riesgo propio."""
    w = plantilla["peso"].to_numpy()
    s = plantilla["segmento"].to_numpy()
    aportacion = w * (mercado.cov @ pesos_segmento(plantilla))[s] + w ** 2 * mercado.propio[s]
    return aportacion / aportacion.sum()
