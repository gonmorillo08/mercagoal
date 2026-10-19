"""
Medidor de fichajes: precio máximo que se puede pagar por un jugador para
obtener beneficio (revenderlo por más de lo pagado) con una probabilidad
elegida.

1. TRASPASOS HISTÓRICOS. Cada cambio de club en el historial de valoraciones
   es un traspaso: valor del jugador justo antes (último valor en el club de
   origen) y valor ~1 año después.

2. EFECTO TRASPASO. La variación de ese año se compara con la ESPERADA para
   un jugador de su posición, edad, año de carrera y tramo de valor (los
   parámetros del módulo de predicción). La diferencia (residuo) es lo que
   el traspaso ha añadido o quitado. Se resume en UNA TABLA de dos entradas:

     - salto de nivel: nivel del club de destino (valor medio de sus
       jugadores de la MISMA POSICIÓN en los 5 años anteriores) frente al
       valor del propio jugador, en 5 categorías (de "jugador muy por encima
       del club" a "club muy por encima del jugador"). Se probó también el
       nivel del club destino frente al del club de origen: predice peor
       fuera de muestra y es demasiado optimista con los jugadores caros;
     - movilidad: cuántas veces ha cambiado ya de club el jugador, en 4
       categorías (de "ningún cambio" a "5 o más").

   El efecto de cada casilla es la media de los residuos de los traspasos
   reales de esa casilla, "encogida" hacia 0 cuando hay pocos casos
   (efecto = suma / (n + K)).

3. SIMULACIÓN. Monte Carlo del módulo de predicción (GBM por posición, edad,
   etapa y tramo) sumando el efecto traspaso a la tendencia del primer año y
   ajustando su volatilidad (los años de traspaso son más inciertos).

4. OFERTA MÁXIMA. Si V_T es el valor al revenderlo dentro de T años,
   P(V_T > precio) = x  <=>  precio = percentil (1 - x) de V_T. Con coste de
   oportunidad anual c, el precio se descuenta: precio / (1 + c)^T.

El valor de mercado se usa como aproximación del precio de reventa (la tabla
no incluye importes de traspaso ni salarios).

Modelo anterior (regresión ridge + historiales de ligas y clubes): copia en
_respaldos/modelo_adaptabilidad_2026-10-07.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from prediccion import modelos as pm

VENTANA_DIAS = (240, 490)      # "un año después": valoración más cercana a 365 días en esta ventana
MAX_HUECO_DIAS = 365           # último valor en origen como mucho un año antes del primero en destino
K_SUAVIZADO = 200              # casos "virtuales" a 0 en cada casilla de la tabla (validado: 50–200)
VENTANA_NIVEL = 5              # años con los que se calcula el nivel de clubes y ligas
MIN_VALORACIONES_NIVEL = 10    # por debajo (en su posición), nivel con todas las posiciones
# Bandas de nivel de club (cuantiles del nivel de los clubes en los que juegan
# los jugadores): 7 bandas, la más alta ~5 % de los casos (clubes de élite).
CUANTILES_BANDA = [0.2, 0.4, 0.6, 0.75, 0.87, 0.95]
K_BANDA = 200                  # casos con los que la banda pesa la mitad frente a los parámetros generales
GRUPOS_DETALLADOS = {"posición + edad + etapa + tramo", "posición + edad + tramo", "posición + edad ±1 + tramo"}
POSICIONES = ["Goalkeeper", "Defender", "Midfield", "Attack"]
PROBABILIDADES = np.round(np.arange(0.50, 0.991, 0.01), 2)
DIAS_CESION = 550              # ida y vuelta al mismo club en menos de ~año y medio: cesión

# Salto de nivel: logaritmo de (valor medio del club destino en la posición
# del jugador / valor del jugador). Cortes en 1/3, 1/1,5, 1,5 y 3 veces.
CORTES_SALTO = np.log([1 / 3, 1 / 1.5, 1.5, 3])
SALTOS = ["Jugador muy por encima del club", "Jugador por encima del club", "Nivel parecido",
          "Club por encima del jugador", "Club muy por encima del jugador"]
DESCRIPCION_SALTO = [
    "el jugador vale más del triple que la media del club en su posición",
    "el jugador vale entre 1,5 y 3 veces la media del club en su posición",
    "el jugador vale parecido a la media del club en su posición (±50 %)",
    "la media del club en su posición vale entre 1,5 y 3 veces lo que el jugador",
    "la media del club en su posición vale más del triple que el jugador",
]
SALTO_DESCONOCIDO = SALTOS.index("Nivel parecido")   # club sin valoraciones: sin salto
# Movilidad: cambios de club anteriores del jugador (sin contar los pasos
# entre el filial y el primer equipo del mismo club).
CORTES_MOVILIDAD = [1, 3, 5]
MOVILIDAD = ["Ningún cambio", "1–2 cambios", "3–4 cambios", "5 o más cambios"]


# ---------------------------------------------------------------------------
# Datos: niveles y traspasos
# ---------------------------------------------------------------------------
class Niveles:
    """Nivel de cada club y liga: media del logaritmo del valor de sus
    jugadores de una POSICIÓN en los últimos `VENTANA_NIVEL` años.

    Para un traspaso del año Y se usan los años Y-4 … Y (el nivel que tenían
    entonces, no el de hoy); para un fichaje actual, los 5 últimos años de
    datos. Si el club (o liga) tiene menos de `MIN_VALORACIONES_NIVEL`
    valoraciones de esa posición en la ventana, se usan todas sus posiciones
    en la misma ventana y, si tampoco llega, todo su histórico. Con
    `hasta_anio` se ignoran los años posteriores (validación sin mirar al
    futuro)."""

    def __init__(self, df: pd.DataFrame, hasta_anio: int | None = None):
        d = df if hasta_anio is None else df[df["fecha"].dt.year <= hasta_anio]
        base = pd.DataFrame({
            "anio": d["fecha"].dt.year.to_numpy(),
            "posicion": d["posicion"].to_numpy(),
            "equipo": d["equipo"].to_numpy(),
            "liga": d["liga"].to_numpy(),
            "log_valor": np.log(d["valor_mercado_eur"].to_numpy(dtype=float)),
        })
        self.anio_min, self.anio_max = int(base["anio"].min()), int(base["anio"].max())
        anios = range(self.anio_min, self.anio_max + 1)
        self.tablas: dict[tuple[str, bool], pd.Series] = {}
        for entidad in ("equipo", "liga"):
            for por_posicion in (True, False):
                claves = [entidad, "posicion"] if por_posicion else [entidad]
                agregado = base.groupby(claves + ["anio"])["log_valor"].agg(["sum", "count"])
                # Suma y nº de valoraciones en la ventana móvil de 5 años
                # (suma acumulada por años menos la de 5 años antes).
                ventana = {}
                for c in ("sum", "count"):
                    tabla = agregado[c].unstack("anio", fill_value=0).reindex(columns=anios, fill_value=0)
                    acumulado = np.cumsum(tabla.to_numpy(dtype=float), axis=1)
                    acumulado[:, VENTANA_NIVEL:] -= acumulado[:, :-VENTANA_NIVEL].copy()
                    ventana[c] = pd.DataFrame(acumulado, index=tabla.index, columns=tabla.columns)
                media = ventana["sum"] / ventana["count"].where(ventana["count"] >= MIN_VALORACIONES_NIVEL)
                self.tablas[(entidad, por_posicion)] = media.stack().rename("nivel")
        # Último respaldo, para clubes casi sin valoraciones en la ventana:
        # su nivel con todo el histórico disponible.
        self.historico = {e: base.groupby(e)["log_valor"].mean() for e in ("equipo", "liga")}

    def nivel(self, entidad: str, nombres, posiciones, anios) -> np.ndarray:
        """Nivel de cada (club o liga, posición, año); NaN si no hay datos."""
        anios = np.clip(np.asarray(anios, dtype=int), self.anio_min, self.anio_max)
        nombres, posiciones = np.asarray(nombres, dtype=object), np.asarray(posiciones, dtype=object)
        propio = self.tablas[(entidad, True)].reindex(pd.MultiIndex.from_arrays([nombres, posiciones, anios]))
        todas = self.tablas[(entidad, False)].reindex(pd.MultiIndex.from_arrays([nombres, anios]))
        historico = pd.Series(nombres).map(self.historico[entidad]).to_numpy(dtype=float)
        nivel = np.where(propio.notna(), propio.to_numpy(), todas.to_numpy()).astype(float)
        return np.where(np.isnan(nivel), historico, nivel)

    def nivel_total(self, entidad: str, nombres, anios) -> np.ndarray:
        """Nivel de cada (club o liga, año) con TODAS las posiciones."""
        anios = np.clip(np.asarray(anios, dtype=int), self.anio_min, self.anio_max)
        nombres = np.asarray(nombres, dtype=object)
        todas = self.tablas[(entidad, False)].reindex(pd.MultiIndex.from_arrays([nombres, anios])).to_numpy(dtype=float)
        historico = pd.Series(nombres).map(self.historico[entidad]).to_numpy(dtype=float)
        return np.where(np.isnan(todas), historico, todas)


_SUFIJO_FILIAL = re.compile(r"\s+(ii|iii|b|c|u\d{2}|sub-?\d{2}|youth|reserves?|primavera|juvenil.*|academy|castilla)$")


def _club_base(equipos: pd.Series) -> pd.Series:
    """Nombre del club sin la marca de filial o cantera ("Real Madrid II" ->
    "real madrid", "Jong Ajax" -> "ajax")."""
    nombres = equipos.astype(str).str.lower().str.strip()
    return nombres.str.replace(r"^jong\s+", "", regex=True).str.replace(_SUFIJO_FILIAL, "", regex=True)


def _entre_filial_y_primer_equipo(ev: pd.DataFrame) -> np.ndarray:
    """Cambios entre el filial y el primer equipo del mismo club: no son
    fichajes (y distorsionan el historial: subir del filial multiplica el valor)."""
    return ((_club_base(ev["equipo_origen"]) == _club_base(ev["equipo_destino"]))
            & (ev["equipo_origen"] != ev["equipo_destino"])).to_numpy(dtype=bool)


def _cesiones(ev: pd.DataFrame) -> np.ndarray:
    """Probables cesiones: ida a un club y vuelta al de origen en menos de
    `DIAS_CESION` días (se marcan la ida y la vuelta)."""
    jugador = ev.groupby("id_player")
    siguiente_destino, siguiente_fecha = jugador["equipo_destino"].shift(-1), jugador["fecha"].shift(-1)
    anterior_origen, anterior_fecha = jugador["equipo_origen"].shift(1), jugador["fecha"].shift(1)
    ida = (siguiente_destino == ev["equipo_origen"]) & ((siguiente_fecha - ev["fecha"]).dt.days <= DIAS_CESION)
    vuelta = (anterior_origen == ev["equipo_destino"]) & ((ev["fecha"] - anterior_fecha).dt.days <= DIAS_CESION)
    return (ida.fillna(False) | vuelta.fillna(False)).to_numpy(dtype=bool)


def _cambios_reales(d: pd.DataFrame) -> np.ndarray:
    """Filas de `d` (ordenado por jugador y fecha) en las que el jugador
    aparece en un club distinto al de su valoración anterior, sin contar los
    pasos entre el filial y el primer equipo del mismo club."""
    codigos, unicos = pd.factorize(d["equipo"].astype(str))
    base = _club_base(pd.Series(unicos)).to_numpy()[codigos]
    ids = d["id_player"].to_numpy()
    cambio = np.zeros(len(d), dtype=bool)
    cambio[1:] = (ids[1:] == ids[:-1]) & (base[1:] != base[:-1])
    return cambio


def cambios_de_club(df: pd.DataFrame) -> pd.Series:
    """Nº de cambios de club de cada jugador en todo su historial (su
    movilidad si fichase hoy)."""
    d = df.sort_values(["id_player", "fecha"])
    return pd.Series(_cambios_reales(d).astype(int)).groupby(d["id_player"].to_numpy()).sum()


def detectar_traspasos(df: pd.DataFrame) -> pd.DataFrame:
    """Un traspaso por cada cambio de club entre dos valoraciones seguidas
    del mismo jugador, con su valor antes y ~1 año después (`log_ret`:
    variación logarítmica anualizada) y los cambios de club que llevaba el
    jugador hasta entonces (`cambios_previos`). Los traspasos demasiado
    recientes para tener ese valor quedan con `log_ret` NaN."""
    d = df.sort_values(["id_player", "fecha"]).reset_index(drop=True)
    cambio = d["id_player"].eq(d["id_player"].shift()) & d["equipo"].ne(d["equipo"].shift())
    i = np.flatnonzero(cambio.to_numpy())
    antes, despues = d.iloc[i - 1], d.iloc[i]
    primer_anio = d.groupby("id_player")["fecha"].transform("min").dt.year.to_numpy()
    reales = _cambios_reales(d).astype(int)
    acumulados = pd.Series(reales).groupby(d["id_player"].to_numpy()).cumsum().to_numpy()
    ev = pd.DataFrame({
        "id_player": despues["id_player"].to_numpy(),
        "nombre": despues["nombre"].to_numpy(),
        "nacionalidad": despues["nacionalidad"].to_numpy(),
        "posicion": despues["posicion"].to_numpy(),
        "altura_cm": despues["altura_cm"].to_numpy(dtype=float),
        "edad": antes["edad"].to_numpy(dtype=float),
        "etapa": antes["fecha"].dt.year.to_numpy() - primer_anio[i - 1],
        "fecha_origen": antes["fecha"].to_numpy().astype("datetime64[ns]"),
        "fecha": despues["fecha"].to_numpy().astype("datetime64[ns]"),
        "equipo_origen": antes["equipo"].to_numpy(),
        "liga_origen": antes["liga"].to_numpy(),
        "equipo_destino": despues["equipo"].to_numpy(),
        "liga_destino": despues["liga"].to_numpy(),
        "valor0": antes["valor_mercado_eur"].to_numpy(dtype=float),
        "cambios_previos": acumulados[i] - reales[i],
    })
    ev["cesion"], ev["filial"] = _cesiones(ev), _entre_filial_y_primer_equipo(ev)
    ev = ev[(ev["fecha"] - ev["fecha_origen"]).dt.days <= MAX_HUECO_DIAS].reset_index(drop=True)
    ev["n"] = np.arange(len(ev))

    # Valor ~1 año después: la valoración más cercana a fecha_origen + 365 días.
    objetivo = ev[["n", "id_player", "fecha_origen"]].assign(
        objetivo=ev["fecha_origen"] + pd.Timedelta(days=365)).sort_values("objetivo")
    valores = (d[["id_player", "fecha", "valor_mercado_eur", "equipo"]]
               .assign(fecha=d["fecha"].astype("datetime64[ns]"))
               .rename(columns={"fecha": "fecha1", "valor_mercado_eur": "valor1", "equipo": "equipo1"})
               .sort_values("fecha1"))
    unido = pd.merge_asof(objetivo, valores, left_on="objetivo", right_on="fecha1", by="id_player",
                          direction="nearest").set_index("n").reindex(ev["n"])
    dias = (unido["fecha1"] - unido["fecha_origen"]).dt.days.to_numpy()
    valido = (dias >= VENTANA_DIAS[0]) & (dias <= VENTANA_DIAS[1]) & (unido["fecha1"].to_numpy() >= ev["fecha"].to_numpy())
    ev["fecha1"] = unido["fecha1"].to_numpy()
    ev["valor1"] = np.where(valido, unido["valor1"].to_numpy(dtype=float), np.nan)
    ev["log_ret"] = np.where(valido, np.log(ev["valor1"] / ev["valor0"]) * 365 / np.where(valido, dias, 365), np.nan)
    ev["anio"] = ev["fecha_origen"].dt.year
    ev["tramo"] = pm.tramo_de(ev["valor0"])
    return ev.drop(columns="n")


def _combinar(general: pm.ParametrosCarrera, banda: pm.ParametrosCarrera) -> pm.ParametrosCarrera:
    """Parámetros de la banda de clubes mezclados con los generales: la banda
    solo pesa en los grupos en los que tiene datos de su posición y edad, y
    tanto más cuantos más casos tenga (peso = n / (n + K_BANDA)); en el resto
    se usan los generales. Separar del todo por bandas empeoraba el modelo
    (grupos demasiado pequeños); mezclarlas así lo mejora."""
    combinado = copy.copy(general)
    detallado = np.isin(banda.nivel.astype(str), list(GRUPOS_DETALLADOS))
    peso = np.where(detallado, banda.n / (banda.n + K_BANDA), 0.0)
    combinado.mu = peso * banda.mu + (1 - peso) * general.mu
    combinado.sigma = np.sqrt(peso * banda.sigma ** 2 + (1 - peso) * general.sigma ** 2)
    combinado.peso_banda = peso
    return combinado


class ParametrosPorNivel:
    """Parámetros del módulo de predicción por posición y BANDA DE NIVEL DEL
    CLUB en el que juega el jugador: un fichaje del Real Madrid se compara con
    jugadores de clubes de valor parecido, no con los de la segunda división
    coreana.

    El nivel del club es el valor medio (geométrico) de su plantilla en los 5
    años anteriores (`Niveles`); las bandas se cortan por cuantiles
    (`CUANTILES_BANDA`), más finas en la parte alta."""

    def __init__(self, retornos: pd.DataFrame, niveles: Niveles):
        self.niveles = niveles
        nivel = niveles.nivel_total("equipo", retornos["equipo"], retornos["anio"])
        reciente = (retornos["anio"] > retornos["anio"].max() - pm.VENTANA_ANIOS).to_numpy()
        self.bordes = np.nanquantile(nivel[reciente], CUANTILES_BANDA)
        banda = self.banda_de(nivel)
        posiciones = [*POSICIONES, None]
        self.general = {p: pm.ParametrosCarrera(retornos, p) for p in posiciones}
        self.por_banda = {}
        for b in range(len(self.bordes) + 1):
            sub = retornos[banda == b]
            for p in posiciones:
                self.por_banda[(b, p)] = _combinar(self.general[p], pm.ParametrosCarrera(sub, p))

    def banda_de(self, nivel_club) -> np.ndarray:
        """Banda de cada nivel de club (-1 si no se conoce: parámetros generales)."""
        nivel_club = np.asarray(nivel_club, dtype=float)
        return np.where(np.isnan(nivel_club), -1, np.searchsorted(self.bordes, np.nan_to_num(nivel_club), side="right"))

    def banda_club(self, equipos, anios) -> np.ndarray:
        return self.banda_de(self.niveles.nivel_total("equipo", equipos, anios))

    def para(self, banda: int, posicion) -> pm.ParametrosCarrera:
        posicion = posicion if posicion in POSICIONES else None
        return self.por_banda.get((int(banda), posicion), self.general[posicion])

    def describir(self, banda: int) -> str:
        """Texto de la banda: número y rango de valor medio de sus clubes."""
        if banda < 0:
            return "todos los clubes"
        limites = [None, *np.exp(self.bordes), None]
        bajo, alto = limites[banda], limites[banda + 1]
        from comun.estilo import euros
        rango = (f"valor medio de plantilla > {euros(bajo)}" if alto is None
                 else f"valor medio de plantilla < {euros(alto)}" if bajo is None
                 else f"valor medio de plantilla entre {euros(bajo)} y {euros(alto)}")
        return f"banda {banda + 1} de {len(self.bordes) + 1} ({rango})"


def fiables(traspasos: pd.DataFrame) -> pd.DataFrame:
    """Traspasos que son fichajes de verdad: sin cesiones (ida y vuelta al
    mismo club) ni cambios entre el filial y el primer equipo."""
    return traspasos[~(traspasos["cesion"] | traspasos["filial"])]


def entrenamiento(traspasos: pd.DataFrame) -> pd.DataFrame:
    """Traspasos con los que se calcula la tabla del efecto traspaso: los
    fiables de los últimos `VENTANA_ANIOS` años, igual que los parámetros de
    la predicción (si no, el efecto de traspasos de hace 20 años se mediría
    contra el mercado de hoy)."""
    recientes = traspasos[traspasos["anio"] > traspasos["anio"].max() - pm.VENTANA_ANIOS]
    return fiables(recientes).reset_index(drop=True)


def esperado(ev: pd.DataFrame, params: ParametrosPorNivel) -> tuple[np.ndarray, np.ndarray]:
    """Tendencia y volatilidad ESPERADAS de cada traspaso sin tener en cuenta
    el traspaso: las del grupo del jugador (posición, edad, etapa, tramo) en
    clubes del nivel del club de destino."""
    mu, sigma = np.zeros(len(ev)), np.zeros(len(ev))
    anio = ev["anio"].to_numpy() if "anio" in ev else np.full(len(ev), params.niveles.anio_max)
    banda = params.banda_club(ev["equipo_destino"], anio)
    posicion = ev["posicion"].fillna("").astype(str).to_numpy()
    posicion = np.where(np.isin(posicion, POSICIONES), posicion, "")
    for b in np.unique(banda):
        for p in np.unique(posicion):
            filas = (banda == b) & (posicion == p)
            if filas.any():
                sub = ev[filas]
                par = params.para(b, p or None)
                a, e, t = par.indices(sub["edad"].to_numpy(dtype=float), sub["etapa"], sub["tramo"])
                mu[filas], sigma[filas] = par.mu[a, e, t], par.sigma[a, e, t]
    return mu, sigma


# ---------------------------------------------------------------------------
# Modelo del efecto traspaso: tabla salto de nivel × movilidad
# ---------------------------------------------------------------------------
@dataclass
class ModeloTraspaso:
    niveles: Niveles
    efecto: np.ndarray           # (salto, movilidad): residuo medio encogido de la casilla
    n: np.ndarray                # (salto, movilidad): traspasos de la casilla
    factor_sigma: np.ndarray     # (salto,): volatilidad del año del traspaso respecto a la de su grupo
    flujos: pd.Series            # (liga origen, liga destino) -> cuota de las salidas de esa liga (informativo)
    k: float = K_SUAVIZADO


def salto_de_nivel(ev: pd.DataFrame, niveles: Niveles) -> tuple[np.ndarray, np.ndarray]:
    """Categoría de salto de nivel de cada traspaso (índice de `SALTOS`) y la
    diferencia de la que sale: logaritmo de (valor medio del club destino en
    la posición del jugador / valor del jugador). Sin año: fichaje actual."""
    anio = ev["anio"].to_numpy() if "anio" in ev else np.full(len(ev), niveles.anio_max)
    club = niveles.nivel("equipo", ev["equipo_destino"], ev["posicion"].to_numpy(), anio)
    diferencia = club - np.log(ev["valor0"].to_numpy(dtype=float))
    salto = np.searchsorted(CORTES_SALTO, np.nan_to_num(diferencia), side="right")
    return np.where(np.isnan(diferencia), SALTO_DESCONOCIDO, salto), diferencia


def movilidad(cambios) -> np.ndarray:
    """Categoría de movilidad (índice de `MOVILIDAD`) según los cambios de
    club anteriores del jugador."""
    return np.searchsorted(CORTES_MOVILIDAD, np.asarray(cambios, dtype=float), side="right")


def flujos_entre_ligas(ev: pd.DataFrame) -> pd.Series:
    """Cuota de los traspasos que salen de cada liga que van a cada otra liga
    (con medio caso añadido para que un flujo raro no cuente como imposible)."""
    pares = ev.groupby(["liga_origen", "liga_destino"]).size()
    salidas = pares.groupby(level="liga_origen").transform("sum")
    return ((pares + 0.5) / (salidas + 1)).rename("cuota")


FIABILIDAD = {"habitual": "Habitual", "poco": "Poco habitual", "inusual": "Inusual"}
CUOTAS_FIABILIDAD = (0.05, 0.01)   # >= 5 % de las salidas de su liga: habitual; < 1 %: inusual


def fiabilidad(flujos: pd.Series, liga_origen, liga_destino) -> np.ndarray:
    """Lo habitual que es cada movimiento entre ligas en el historial: qué
    parte de los traspasos que salen de la liga de origen van a la de destino.
    Quedarse en la misma liga siempre es habitual. Es solo informativo: no
    entra en el cálculo del efecto traspaso."""
    origen = np.asarray(liga_origen, dtype=object)
    destino = np.broadcast_to(np.asarray(liga_destino, dtype=object), origen.shape)   # admite una sola liga destino
    cuota = flujos.reindex(pd.MultiIndex.from_arrays([origen, destino])).fillna(0).to_numpy(dtype=float)
    etiqueta = np.where(cuota >= CUOTAS_FIABILIDAD[0], "habitual",
                        np.where(cuota >= CUOTAS_FIABILIDAD[1], "poco", "inusual"))
    return np.where(origen == destino, "habitual", etiqueta)


def ajustar(ev: pd.DataFrame, residuo: np.ndarray, sigma_grupo: np.ndarray, niveles: Niveles,
            k: float = K_SUAVIZADO) -> ModeloTraspaso:
    """Tabla del efecto traspaso: media de los residuos (variación real −
    esperada) de cada casilla salto de nivel × movilidad, encogida hacia 0."""
    salto, _ = salto_de_nivel(ev, niveles)
    mov = movilidad(ev["cambios_previos"])
    forma = (len(SALTOS), len(MOVILIDAD))
    suma, n = np.zeros(forma), np.zeros(forma)
    np.add.at(suma, (salto, mov), residuo)
    np.add.at(n, (salto, mov), 1)
    efecto = suma / (n + k)
    # Los años de traspaso no son igual de inciertos en todos los saltos: los
    # jugadores que fichan por un club muy por encima de ellos se dispersan más.
    z = (residuo - efecto[salto, mov]) / sigma_grupo
    factor = np.array([z[salto == s].std() if (salto == s).sum() > 1 else z.std() for s in range(forma[0])])
    return ModeloTraspaso(niveles=niveles, efecto=efecto, n=n.astype(int), factor_sigma=factor,
                          flujos=flujos_entre_ligas(ev), k=k)


def predecir(modelo: ModeloTraspaso, ev: pd.DataFrame) -> pd.DataFrame:
    """Casilla de cada traspaso (salto de nivel y movilidad) y lo que dice la
    tabla para ella: efecto sobre la variación logarítmica del primer año, nº
    de traspasos históricos y factor de volatilidad. `ev` necesita:
    equipo_destino, posicion, valor0 y cambios_previos."""
    salto, diferencia = salto_de_nivel(ev, modelo.niveles)
    mov = movilidad(ev["cambios_previos"])
    return pd.DataFrame({
        "salto": salto, "movilidad": mov, "diferencia": diferencia,
        "efecto": modelo.efecto[salto, mov], "n": modelo.n[salto, mov], "factor": modelo.factor_sigma[salto],
    })


# ---------------------------------------------------------------------------
# Simulación y oferta máxima
# ---------------------------------------------------------------------------
def simular_fichaje(v0: float, edad0: int, etapa0: int, anios: int, params: pm.ParametrosCarrera,
                    efecto: float, factor_sigma: float, n: int, semilla: int = 0) -> np.ndarray:
    """Monte Carlo del módulo de predicción con el efecto traspaso en el
    primer año (tendencia + efecto, volatilidad × factor)."""
    rng = np.random.default_rng(semilla)
    log_v = np.full(n, np.log(v0))
    caminos = [np.full(n, v0)]
    for y in range(anios):
        a, e, t = params.indices(np.full(n, edad0 + y), np.full(n, etapa0 + y), pm.tramo_de(np.exp(log_v)))
        mu, sigma = params.mu[a, e, t], params.sigma[a, e, t]
        if y == 0:
            mu, sigma = mu + efecto, sigma * factor_sigma
        log_v = np.maximum(log_v + mu + sigma * rng.standard_normal(n), np.log(pm.VALOR_MINIMO))
        caminos.append(np.exp(log_v))
    return np.column_stack(caminos)


def oferta_maxima(final: np.ndarray, probabilidad, coste_anual: float, anios: int):
    """Precio máximo para que P(valor de reventa > precio × (1 + c)^T) sea la
    probabilidad pedida: percentil (1 − x) del valor final, descontado."""
    return np.percentile(final, 100 * (1 - np.asarray(probabilidad))) / (1 + coste_anual) ** anios


# ---------------------------------------------------------------------------
# Validación fuera de muestra
# ---------------------------------------------------------------------------
def validar(df: pd.DataFrame, ev: pd.DataFrame, retornos: pd.DataFrame, anio_corte: int = 2018,
            k: float = K_SUAVIZADO) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Calcula la tabla con los traspasos que empiezan hasta `anio_corte` y la
    evalúa con los fichajes posteriores (a 1 año). Devuelve métricas,
    calibración de la oferta máxima (¿se obtiene beneficio en el x % de los
    casos?) e info."""
    con_dato = ev[ev["log_ret"].notna()]
    train = entrenamiento(con_dato[con_dato["anio"] <= anio_corte])
    test = fiables(con_dato[con_dato["anio"] > anio_corte]).reset_index(drop=True)
    niveles = Niveles(df, anio_corte)
    params = ParametrosPorNivel(retornos[retornos["anio"] <= anio_corte], niveles)
    mu_tr, sd_tr = esperado(train, params)
    modelo = ajustar(train, train["log_ret"].to_numpy() - mu_tr, sd_tr, niveles, k)
    mu_te, sd_te = esperado(test, params)
    prediccion = predecir(modelo, test)
    efecto = prediccion["efecto"].to_numpy()
    r = test["log_ret"].to_numpy()
    elite = params.banda_club(test["equipo_destino"], test["anio"]) == len(params.bordes)

    def metricas(nombre, centro, sigma):
        return {
            "modelo": nombre,
            "error_medio": float(np.mean(np.abs(r - centro))),
            "nll": float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * ((r - centro) / sigma) ** 2)),
            "correlacion": float(np.corrcoef(centro - mu_te, r - mu_te)[0, 1]) if np.std(centro - mu_te) > 0 else np.nan,
        }

    sd_ajustada = sd_te * prediccion["factor"].to_numpy()
    filas = [
        metricas("Predicción sin efecto traspaso", mu_te, sd_te),
        metricas("Predicción + efecto traspaso", mu_te + efecto, sd_ajustada),
    ]
    calibracion = []
    for x in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        z = norm.ppf(1 - x)
        con_efecto = r > mu_te + efecto + sd_ajustada * z
        calibracion.append({
            "probabilidad_pedida": x,
            "sin_efecto": float(np.mean(r > mu_te + sd_te * z)),
            "con_efecto": float(np.mean(con_efecto)),
            "con_efecto_elite": float(np.mean(con_efecto[elite])),
        })
    info = {"n_train": len(train), "n_test": len(test), "anio_corte": anio_corte, "n_elite": int(elite.sum()),
            "elite": params.describir(len(params.bordes))}
    return pd.DataFrame(filas), pd.DataFrame(calibracion), info
