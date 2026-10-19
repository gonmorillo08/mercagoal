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
   el traspaso ha añadido o quitado. Se explica con un modelo aditivo:

     - perfil del jugador: edad, altura, posición, nacionalidad y si pasa a
       jugar en la liga de su país (regresión ridge);
     - nivel de las ligas y clubes de origen y destino (valor medio de sus
       jugadores de la MISMA POSICIÓN en los 5 años anteriores al traspaso) y
       lo que vale el jugador respecto al club destino (misma regresión);
     - historial liga origen -> liga destino, club destino y club origen ->
       club destino: media de los residuos de los traspasos reales de ese
       grupo, "encogida" hacia 0 cuando hay pocos casos
       (efecto = suma / (n + K)): con 3 casos apenas cuenta; con 300, casi
       todo.

3. SIMULACIÓN. Monte Carlo del módulo de predicción (GBM por posición, edad,
   etapa y tramo) sumando el efecto traspaso a la tendencia del primer año y
   ajustando su volatilidad (los años de traspaso son más inciertos).

4. OFERTA MÁXIMA. Si V_T es el valor al revenderlo dentro de T años,
   P(V_T > precio) = x  <=>  precio = percentil (1 - x) de V_T. Con coste de
   oportunidad anual c, el precio se descuenta: precio / (1 + c)^T.

El valor de mercado se usa como aproximación del precio de reventa (la tabla
no incluye importes de traspaso ni salarios).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from prediccion import modelos as pm

VENTANA_DIAS = (240, 490)      # "un año después": valoración más cercana a 365 días en esta ventana
MAX_HUECO_DIAS = 365           # último valor en origen como mucho un año antes del primero en destino
K_SUAVIZADO = 200              # casos "virtuales" a 0 en el historial de cada grupo (validado: 30–1000)
LAMBDA_RIDGE = 20_000.0         # penalización ridge (validada fuera de muestra: 10–500.000)
VENTANA_NIVEL = 5              # años con los que se calcula el nivel de clubes y ligas
MIN_VALORACIONES_NIVEL = 10    # por debajo (en su posición), nivel con todas las posiciones
# Bandas de nivel de club (cuantiles del nivel de los clubes en los que juegan
# los jugadores): 7 bandas, la más alta ~5 % de los casos (clubes de élite).
CUANTILES_BANDA = [0.2, 0.4, 0.6, 0.75, 0.87, 0.95]
K_BANDA = 200                  # casos con los que la banda pesa la mitad frente a los parámetros generales
MIN_TRASPASOS_FACTOR = 500     # traspasos de una banda para estimar su propio factor de volatilidad
GRUPOS_DETALLADOS = {"posición + edad + etapa + tramo", "posición + edad + tramo", "posición + edad ±1 + tramo"}
N_NACIONALIDADES = 25        # nacionalidades con variable propia; el resto, "otra"
POSICIONES = ["Goalkeeper", "Defender", "Midfield", "Attack"]
PROBABILIDADES = np.round(np.arange(0.50, 0.991, 0.01), 2)

PERFIL = "Perfil del jugador"
NIVEL = "Nivel de ligas y clubes"
ADAPTACION = "Adaptación al destino"
HIST_LIGA_DESTINO = "Historial de la liga destino"
HIST_LIGAS = "Historial liga origen → destino"
HIST_NACIONALIDAD = "Historial nacionalidad → liga destino"
HIST_DESTINO = "Historial del club destino"
HIST_CLUBES = "Historial club origen → destino"
COMPONENTES = ["Traspaso medio", PERFIL, NIVEL, HIST_LIGAS, HIST_DESTINO, HIST_CLUBES]   # modelo clásico

# ---------------------------------------------------------------------------
# Versiones del modelo de traspaso. Para volver al modelo anterior basta con
# poner VERSION_MODELO = "clasico" (reproduce exactamente el modelo previo a
# la adaptabilidad). Copia del código anterior: _respaldos/modelo_clasico_2026-09-28.
# ---------------------------------------------------------------------------
DIAS_CESION = 550              # ida y vuelta al mismo club en menos de ~año y medio: cesión
CUOTA_MINIMA = 1e-4           # cuota de un flujo entre ligas nunca visto
LAMBDA_ADAPTACION = 2_000.0   # penalización (menor) de las variables de nivel y adaptación
VERSIONES = {
    "clasico": dict(limpiar=False, prueba_fiable=False, nivel_asimetrico=False, fiabilidad=False,
                    hist_liga_destino=False, hist_nacionalidad=False, lam_adaptacion=None),
    # Más peso a la adaptabilidad: bajar de nivel de liga/club se penaliza por
    # separado de subir, los movimientos poco habituales (p. ej. LaLiga ->
    # China) cuentan como tales, y se usa el historial de TODAS las llegadas
    # a la liga destino y el de los jugadores de su nacionalidad en ella, con
    # menos penalización para que pesen más. Se valida y se muestra solo con
    # traspasos fiables (sin cesiones ni cambios con el filial); se ENTRENA
    # con todos porque quitarlos empeoraba el modelo fuera de muestra.
    "adaptabilidad": dict(limpiar=False, prueba_fiable=True, nivel_asimetrico=True, fiabilidad=True,
                          hist_liga_destino=True, hist_nacionalidad=True, lam_adaptacion=LAMBDA_ADAPTACION),
}
VERSION_MODELO = "adaptabilidad"


def ajustes_version(version: str | None = None) -> dict:
    return dict(VERSIONES[version or VERSION_MODELO])


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


def detectar_traspasos(df: pd.DataFrame) -> pd.DataFrame:
    """Un traspaso por cada cambio de club entre dos valoraciones seguidas
    del mismo jugador, con su valor antes y ~1 año después (`log_ret`:
    variación logarítmica anualizada). Los traspasos demasiado recientes para
    tener ese valor quedan con `log_ret` NaN."""
    d = df.sort_values(["id_player", "fecha"]).reset_index(drop=True)
    cambio = d["id_player"].eq(d["id_player"].shift()) & d["equipo"].ne(d["equipo"].shift())
    i = np.flatnonzero(cambio.to_numpy())
    antes, despues = d.iloc[i - 1], d.iloc[i]
    primer_anio = d.groupby("id_player")["fecha"].transform("min").dt.year.to_numpy()
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


def entrenamiento(traspasos: pd.DataFrame, ajustes: dict | None = None) -> pd.DataFrame:
    """Traspasos con los que se ajusta el efecto traspaso: los de los últimos
    `VENTANA_ANIOS` años, igual que los parámetros de la predicción (si no,
    el efecto de traspasos de hace 20 años se mediría contra el mercado de
    hoy). Con la versión "adaptabilidad", solo traspasos fiables: sin
    cesiones ni cambios entre filial y primer equipo."""
    ajustes = ajustes or ajustes_version()
    elegidos = traspasos["anio"] > traspasos["anio"].max() - pm.VENTANA_ANIOS
    if ajustes["limpiar"]:
        elegidos &= ~(traspasos["cesion"] | traspasos["filial"])
    return traspasos[elegidos].reset_index(drop=True)


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
# Modelo del efecto traspaso
# ---------------------------------------------------------------------------
@dataclass
class ModeloTraspaso:
    nacionalidades: list[str]
    niveles: Niveles
    columnas: list[str] = field(default_factory=list)
    grupo_columna: list[str] = field(default_factory=list)
    media: np.ndarray | None = None
    desv: np.ndarray | None = None
    intercepto: float = 0.0
    beta: np.ndarray | None = None
    historial: dict = field(default_factory=dict)   # componente -> DataFrame (efecto, n) indexado por grupo
    factor_sigma: float = 1.0
    factor_por_banda: dict = field(default_factory=dict)   # banda del club destino -> factor de volatilidad
    k: float = K_SUAVIZADO
    limites: dict = field(default_factory=dict)      # variable continua -> (percentil 1, percentil 99)
    ajustes: dict = field(default_factory=ajustes_version)
    flujos: pd.Series | None = None                  # (liga origen, liga destino) -> cuota de las salidas de esa liga

    def factor(self, banda: int) -> float:
        """Volatilidad del año del traspaso respecto a la de su grupo, según la
        banda de nivel del club de destino (global si la banda tiene pocos casos)."""
        return self.factor_por_banda.get(int(banda), self.factor_sigma)


CONTINUAS = ["Edad", "Edad²", "Altura", "Liga destino − liga origen", "Club destino − club origen",
             "Valor del jugador − nivel del club destino", "Sube de nivel de liga", "Baja de nivel de liga",
             "Sube de nivel de club", "Baja de nivel de club", "Traspaso poco habitual"]

CLAVES_HISTORIAL = {
    HIST_LIGAS: ["liga_origen", "liga_destino"],
    HIST_DESTINO: ["equipo_destino"],
    HIST_CLUBES: ["equipo_origen", "equipo_destino"],
}


def claves_historial(ajustes: dict) -> dict:
    """Historiales que usa cada versión, en el orden en que se ajustan (de lo
    general a lo particular: cada uno explica lo que dejó el anterior)."""
    claves = {}
    if ajustes["hist_liga_destino"]:
        claves[HIST_LIGA_DESTINO] = ["liga_destino"]
    claves[HIST_LIGAS] = ["liga_origen", "liga_destino"]
    if ajustes["hist_nacionalidad"]:
        claves[HIST_NACIONALIDAD] = ["nacionalidad", "liga_destino"]
    claves[HIST_DESTINO] = ["equipo_destino"]
    claves[HIST_CLUBES] = ["equipo_origen", "equipo_destino"]
    return claves


def componentes(modelo: "ModeloTraspaso") -> list[str]:
    """Componentes del efecto traspaso de un modelo ajustado."""
    grupos = ["Traspaso medio", PERFIL, NIVEL]
    if modelo.ajustes["nivel_asimetrico"] or modelo.ajustes["fiabilidad"]:
        grupos.append(ADAPTACION)
    return grupos + list(claves_historial(modelo.ajustes))


def _matriz(ev: pd.DataFrame, modelo: ModeloTraspaso) -> tuple[np.ndarray, list[str], list[str]]:
    """Variables explicativas de cada traspaso y a qué componente pertenecen."""
    edad = ev["edad"].to_numpy(dtype=float) - 25
    altura = ev["altura_cm"].to_numpy(dtype=float)
    # Nivel de clubes y ligas en la posición del jugador y en los 5 años que
    # terminan en el del traspaso (sin año: fichaje actual, últimos 5 años).
    anio = ev["anio"].to_numpy() if "anio" in ev else np.full(len(ev), modelo.niveles.anio_max)
    posicion = ev["posicion"].to_numpy()
    liga_o = modelo.niveles.nivel("liga", ev["liga_origen"], posicion, anio)
    liga_d = modelo.niveles.nivel("liga", ev["liga_destino"], posicion, anio)
    club_o = modelo.niveles.nivel("equipo", ev["equipo_origen"], posicion, anio)
    club_d = modelo.niveles.nivel("equipo", ev["equipo_destino"], posicion, anio)
    columnas = {
        "Edad": (edad, PERFIL),
        "Edad²": (edad ** 2, PERFIL),
        "Altura": (np.nan_to_num(altura - 182, nan=0.0), PERFIL),
        "Altura desconocida": (np.isnan(altura).astype(float), PERFIL),
        "Juega en la liga de su país": ((ev["nacionalidad"] == ev["liga_destino"]).to_numpy(dtype=float), PERFIL),
    }
    for p in POSICIONES:
        columnas[f"Posición: {p}"] = ((ev["posicion"] == p).to_numpy(dtype=float), PERFIL)
    for n in modelo.nacionalidades:
        columnas[f"Nacionalidad: {n}"] = ((ev["nacionalidad"] == n).to_numpy(dtype=float), PERFIL)
    dif_liga, dif_club = np.nan_to_num(liga_d - liga_o), np.nan_to_num(club_d - club_o)
    columnas["Cambia de liga"] = ((ev["liga_origen"] != ev["liga_destino"]).to_numpy(dtype=float), NIVEL)
    if modelo.ajustes["nivel_asimetrico"]:
        # Subir y bajar de nivel no son simétricos: bajar a una liga o club
        # mucho más débil hunde el valor más de lo que lo sube el movimiento
        # contrario.
        columnas.update({
            "Sube de nivel de liga": (np.maximum(dif_liga, 0), ADAPTACION),
            "Baja de nivel de liga": (np.maximum(-dif_liga, 0), ADAPTACION),
            "Sube de nivel de club": (np.maximum(dif_club, 0), ADAPTACION),
            "Baja de nivel de club": (np.maximum(-dif_club, 0), ADAPTACION),
        })
    else:
        columnas.update({
            "Liga destino − liga origen": (dif_liga, NIVEL),
            "Club destino − club origen": (dif_club, NIVEL),
        })
    columnas["Valor del jugador − nivel del club destino"] = (
        np.nan_to_num(np.log(ev["valor0"].to_numpy(dtype=float)) - club_d), NIVEL)
    if modelo.ajustes["fiabilidad"] and modelo.flujos is not None:
        # Fiabilidad: qué parte de los traspasos que salen de la liga de
        # origen van a la de destino (LaLiga -> Premier es habitual; LaLiga ->
        # China, no). −log10 de esa cuota: 0 = muy habitual, 4 = casi inédito.
        cuota = pd.MultiIndex.from_arrays([ev["liga_origen"].to_numpy(dtype=object),
                                           ev["liga_destino"].to_numpy(dtype=object)])
        cuota = modelo.flujos.reindex(cuota).to_numpy(dtype=float)
        columnas["Traspaso poco habitual"] = (-np.log10(np.nan_to_num(cuota, nan=CUOTA_MINIMA).clip(CUOTA_MINIMA)),
                                              ADAPTACION)
    X = np.column_stack([v for v, _ in columnas.values()])
    nombres = list(columnas)
    if modelo.limites:
        # Las variables continuas no salen del rango visto al ajustar: una
        # relación lineal extrapolada (p. ej. un jugador de 200 M€ muy por
        # encima del nivel de cualquier club) daría efectos irreales.
        for nombre, (bajo, alto) in modelo.limites.items():
            j = nombres.index(nombre)
            X[:, j] = np.clip(X[:, j], bajo, alto)
    return X, nombres, [g for _, g in columnas.values()]


def flujos_entre_ligas(ev: pd.DataFrame) -> pd.Series:
    """Cuota de los traspasos que salen de cada liga que van a cada otra liga
    (con medio caso añadido para que un flujo raro no cuente como imposible)."""
    pares = ev.groupby(["liga_origen", "liga_destino"]).size()
    salidas = pares.groupby(level="liga_origen").transform("sum")
    return ((pares + 0.5) / (salidas + 1)).rename("cuota")


FIABILIDAD = {"habitual": "Habitual", "poco": "Poco habitual", "inusual": "Inusual"}
CUOTAS_FIABILIDAD = (0.05, 0.01)   # >= 5 % de las salidas de su liga: habitual; < 1 %: inusual


def fiabilidad(flujos: pd.Series, liga_origen, liga_destino) -> np.ndarray:
    """Fiabilidad de cada movimiento entre ligas según lo habitual que es en
    el historial: qué parte de los traspasos que salen de la liga de origen
    van a la de destino. Quedarse en la misma liga siempre es habitual."""
    origen = np.asarray(liga_origen, dtype=object)
    destino = np.broadcast_to(np.asarray(liga_destino, dtype=object), origen.shape)   # admite una sola liga destino
    cuota = flujos.reindex(pd.MultiIndex.from_arrays([origen, destino])).fillna(0).to_numpy(dtype=float)
    etiqueta = np.where(cuota >= CUOTAS_FIABILIDAD[0], "habitual",
                        np.where(cuota >= CUOTAS_FIABILIDAD[1], "poco", "inusual"))
    return np.where(origen == destino, "habitual", etiqueta)


def ajustar(ev: pd.DataFrame, residuo: np.ndarray, sigma_grupo: np.ndarray, niveles: Niveles,
            k: float = K_SUAVIZADO, lam: float = LAMBDA_RIDGE, banda: np.ndarray | None = None,
            ajustes: dict | None = None) -> ModeloTraspaso:
    """Ajusta el efecto traspaso sobre los residuos (variación real − esperada).
    `banda`: banda de nivel del club de destino de cada traspaso, para estimar
    la volatilidad extra del traspaso por banda. `ajustes`: versión del modelo
    (por defecto, `VERSION_MODELO`)."""
    ajustes = ajustes or ajustes_version()
    nacionalidades = ev["nacionalidad"].value_counts().head(N_NACIONALIDADES).index.tolist()
    modelo = ModeloTraspaso(nacionalidades=nacionalidades, niveles=niveles, k=k, ajustes=ajustes)
    # Flujos entre ligas: variable del modelo (versión con adaptabilidad) y
    # etiqueta de fiabilidad del traspaso (todas las versiones).
    modelo.flujos = flujos_entre_ligas(ev)
    X, columnas, _ = _matriz(ev, modelo)
    modelo.limites = {c: tuple(np.percentile(X[:, columnas.index(c)], [1, 99])) for c in CONTINUAS
                      if c in columnas}
    X, modelo.columnas, modelo.grupo_columna = _matriz(ev, modelo)
    modelo.media, modelo.desv = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - modelo.media) / modelo.desv
    modelo.intercepto = float(residuo.mean())
    y = residuo - modelo.intercepto
    # Penalización ridge por variable: las de nivel y adaptación, menos
    # penalizadas en la versión con adaptabilidad (pesan más).
    penalizacion = np.full(Z.shape[1], lam)
    if ajustes["lam_adaptacion"] is not None:
        grupos = np.array(modelo.grupo_columna)
        penalizacion[(grupos == NIVEL) | (grupos == ADAPTACION)] = ajustes["lam_adaptacion"]
    modelo.beta = np.linalg.solve(Z.T @ Z + np.diag(penalizacion), Z.T @ y)

    # Historial por grupos, en cascada sobre lo que aún no está explicado.
    resto = y - Z @ modelo.beta
    for componente, claves in claves_historial(ajustes).items():
        g = pd.DataFrame({"r": resto}).join(ev[claves].reset_index(drop=True)).groupby(claves)["r"]
        tabla = pd.DataFrame({"n": g.size(), "suma": g.sum()})
        tabla["efecto"] = tabla["suma"] / (tabla["n"] + k)
        modelo.historial[componente] = tabla[["efecto", "n"]]
        resto = resto - _efecto_grupo(ev, tabla, claves)[0]

    # Los años de traspaso son más volátiles que la media de su grupo.
    z = resto / sigma_grupo
    modelo.factor_sigma = float(np.std(z))
    if banda is not None:
        # En los clubes de élite los resultados tras un fichaje se dispersan
        # más que en su grupo (factor ~1,18 frente a ~0,9 en los modestos): con
        # un único factor, la oferta máxima salía demasiado alta en la élite.
        modelo.factor_por_banda = {int(b): float(np.std(z[banda == b])) for b in np.unique(banda)
                                   if (banda == b).sum() >= MIN_TRASPASOS_FACTOR}
    return modelo


def _efecto_grupo(ev: pd.DataFrame, tabla: pd.DataFrame, claves: list[str]) -> tuple[np.ndarray, np.ndarray]:
    unido = ev[claves].reset_index(drop=True).merge(tabla, left_on=claves, right_index=True, how="left")
    return unido["efecto"].fillna(0).to_numpy(), unido["n"].fillna(0).astype(int).to_numpy()


def predecir(modelo: ModeloTraspaso, ev: pd.DataFrame) -> pd.DataFrame:
    """Efecto traspaso (sobre la variación logarítmica del primer año)
    desglosado por componentes, más el nº de casos de cada historial."""
    X, _, grupos = _matriz(ev, modelo)
    contrib = ((X - modelo.media) / modelo.desv) * modelo.beta
    grupos = np.array(grupos)
    out = pd.DataFrame({
        "Traspaso medio": np.full(len(ev), modelo.intercepto),
        PERFIL: contrib[:, grupos == PERFIL].sum(axis=1),
        NIVEL: contrib[:, grupos == NIVEL].sum(axis=1),
    })
    if ADAPTACION in componentes(modelo):
        out[ADAPTACION] = contrib[:, grupos == ADAPTACION].sum(axis=1)
    for componente, claves in claves_historial(modelo.ajustes).items():
        out[componente], out[f"n · {componente}"] = _efecto_grupo(ev, modelo.historial[componente], claves)
    out["efecto"] = out[componentes(modelo)].sum(axis=1)
    return out


def contribuciones_perfil(modelo: ModeloTraspaso, ev: pd.DataFrame) -> pd.Series:
    """Contribución de cada variable del perfil y del nivel (un solo traspaso)."""
    X, columnas, _ = _matriz(ev, modelo)
    contrib = (((X - modelo.media) / modelo.desv) * modelo.beta)[0]
    return pd.Series(contrib, index=columnas)


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
            k: float = K_SUAVIZADO, lam: float = LAMBDA_RIDGE, ajustes: dict | None = None,
            params: "ParametrosPorNivel | None" = None,
            prueba_fiable: bool | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Ajusta con los traspasos que empiezan hasta `anio_corte` y evalúa con
    los posteriores (a 1 año). Devuelve métricas, calibración de la oferta
    máxima (¿se obtiene beneficio en el x % de los casos?) e info."""
    con_dato = ev[ev["log_ret"].notna()]
    ajustes = ajustes or ajustes_version()
    train = entrenamiento(con_dato[con_dato["anio"] <= anio_corte], ajustes)
    test = con_dato[con_dato["anio"] > anio_corte]
    # Se evalúa con fichajes reales (sin cesiones ni filial) si la versión
    # entrena solo con ellos: son los que interesan al fichar.
    if ajustes["prueba_fiable"] if prueba_fiable is None else prueba_fiable:
        test = test[~(test["cesion"] | test["filial"])]
    test = test.reset_index(drop=True)
    niveles = Niveles(df, anio_corte) if params is None else params.niveles
    params = params or ParametrosPorNivel(retornos[retornos["anio"] <= anio_corte], niveles)
    mu_tr, sd_tr = esperado(train, params)
    modelo = ajustar(train, train["log_ret"].to_numpy() - mu_tr, sd_tr, niveles, k, lam,
                     params.banda_club(train["equipo_destino"], train["anio"]), ajustes)
    mu_te, sd_te = esperado(test, params)
    efecto = predecir(modelo, test)["efecto"].to_numpy()
    r = test["log_ret"].to_numpy()
    banda_te = params.banda_club(test["equipo_destino"], test["anio"])
    elite = banda_te == len(params.bordes)

    def metricas(nombre, centro, sigma):
        return {
            "modelo": nombre,
            "error_medio": float(np.mean(np.abs(r - centro))),
            "nll": float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * ((r - centro) / sigma) ** 2)),
            "correlacion": float(np.corrcoef(centro - mu_te, r - mu_te)[0, 1]) if np.std(centro - mu_te) > 0 else np.nan,
        }

    sd_ajustada = sd_te * np.array([modelo.factor(b) for b in banda_te])
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
