"""
Mejores fichajes: rentabilidad de fichar a cada jugador candidato para un
equipo, combinando los dos módulos anteriores:

  - Medidor de fichajes (`fichajes.modelo`): efecto traspaso de cada
    candidato hacia el equipo comprador (nivel del club frente al valor del
    jugador, cuántas veces ha cambiado ya de club, su edad y la adaptabilidad:
    historial de sus clubes, ligas y nacionalidad).
  - Predicción (`prediccion.modelos`): Monte Carlo de su carrera por
    posición, edad, año de carrera y tramo de valor.

Se simulan todos los candidatos a la vez (vectorizado, por bloques) y de cada
uno se guardan los percentiles de su valor al venderlo. Con ellos, sin volver
a simular, se calculan para cualquier probabilidad x, prima y coste de
oportunidad:

  - oferta máxima: percentil (1 − x) del valor de venta, descontado;
  - margen con probabilidad x: oferta máxima / precio − 1 (con probabilidad
    x, la rentabilidad es al menos esa);
  - beneficio esperado, mediano y con probabilidad x (en euros),
    rentabilidad mediana y probabilidad de beneficio.

El precio se aproxima con el valor de mercado (más una prima opcional): la
tabla no incluye importes reales de traspaso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fichajes import modelo as fm
from prediccion import modelos as pm

PERCENTILES = np.arange(0.5, 100, 0.5)          # rejilla de percentiles guardada por jugador
POSICIONES_ES = {"Goalkeeper": "Portero", "Defender": "Defensa", "Midfield": "Centrocampista",
                 "Attack": "Delantero"}
# El primero es el orden por defecto: dinero generado en euros (no en %).
CRITERIOS = {
    "beneficio_esperado": "Beneficio esperado (€)",
    "beneficio_mediano": "Beneficio mediano (€)",
    "beneficio_asegurado": "Beneficio con la probabilidad elegida (€)",
    "margen": "Margen con la probabilidad elegida (%)",
    "rentabilidad_mediana": "Rentabilidad mediana (%)",
    "prob_beneficio": "Probabilidad de beneficio",
}
CRITERIOS_EUROS = {"beneficio_esperado", "beneficio_mediano", "beneficio_asegurado"}
BLOQUE = 400                                    # jugadores simulados a la vez (memoria acotada)


def simular_percentiles(v0, edad0, etapa0, efecto, factor_sigma, anios: int,
                        params: pm.ParametrosCarrera, n: int, semilla: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo de `fichajes.modelo.simular_fichaje` para muchos jugadores
    a la vez (efecto traspaso y factor de volatilidad de cada uno). Devuelve
    la rejilla de percentiles del valor final (jugadores × `PERCENTILES`) y
    su media."""
    rng = np.random.default_rng(semilla)
    v0, edad0, etapa0, efecto, factor_sigma = (np.asarray(x, dtype=float)
                                               for x in (v0, edad0, etapa0, efecto, factor_sigma))
    rejilla = np.empty((len(v0), len(PERCENTILES)), dtype=np.float32)
    media = np.empty(len(v0))
    for ini in range(0, len(v0), BLOQUE):
        b = slice(ini, ini + BLOQUE)
        log_v = np.repeat(np.log(v0[b])[:, None], n, axis=1)
        for y in range(anios):
            forma = log_v.shape
            a, e, t = params.indices(np.broadcast_to((edad0[b] + y)[:, None], forma),
                                     np.broadcast_to((etapa0[b] + y)[:, None], forma), pm.tramo_de(np.exp(log_v)))
            mu, sigma = params.mu[a, e, t], params.sigma[a, e, t]
            if y == 0:
                mu, sigma = mu + efecto[b][:, None], sigma * factor_sigma[b][:, None]
            log_v = np.maximum(log_v + mu + sigma * rng.standard_normal(forma), np.log(pm.VALOR_MINIMO))
        final = np.exp(log_v)
        rejilla[b] = np.percentile(final, PERCENTILES, axis=1).T
        media[b] = final.mean(axis=1)
    return rejilla, media


def evaluar(candidatos: pd.DataFrame, modelo: fm.ModeloTraspaso, params: fm.ParametrosPorNivel, destino: str,
            liga_destino: str,
            anios: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Efecto traspaso, rejilla de percentiles y media del valor de venta de
    cada candidato si ficha por `destino`. `candidatos` necesita: nacionalidad,
    posicion, edad_actual, etapa, equipo_actual, liga_actual, valor_actual y
    cambios_previos."""
    fila = pd.DataFrame({
        "nacionalidad": candidatos["nacionalidad"].to_numpy(),
        "posicion": candidatos["posicion"].to_numpy(),
        "edad": candidatos["edad_actual"].to_numpy(dtype=float),
        "equipo_origen": candidatos["equipo_actual"].to_numpy(),
        "liga_origen": candidatos["liga_actual"].to_numpy(),
        "equipo_destino": destino,
        "liga_destino": liga_destino,
        "valor0": candidatos["valor_actual"].to_numpy(dtype=float),
        "cambios_previos": candidatos["cambios_previos"].to_numpy(),
    })
    prediccion = fm.predecir(modelo, fila)
    efecto, factor = prediccion["efecto"].to_numpy(), prediccion["factor"].to_numpy()
    rejilla = np.empty((len(candidatos), len(PERCENTILES)), dtype=np.float32)
    media = np.empty(len(candidatos))
    # Evolución tras el fichaje: jugadores de su posición en clubes del nivel
    # del destino. Sin posición (NA) -> parámetros de edad + tramo (clave None).
    banda = int(params.banda_club([destino], [params.niveles.anio_max])[0])
    posicion = candidatos["posicion"].fillna("").astype(str).to_numpy()
    conocida = np.isin(posicion, fm.POSICIONES)
    for p in [*fm.POSICIONES, None]:
        filas = ~conocida if p is None else conocida & (posicion == p)
        if filas.any():
            sub = candidatos[filas]
            rejilla[filas], media[filas] = simular_percentiles(
                sub["valor_actual"], sub["edad_actual"], sub["etapa"], efecto[filas], factor[filas],
                anios, params.para(banda, p), n)
    return efecto, rejilla, media


def metricas(valor_actual, rejilla: np.ndarray, media: np.ndarray, prima: float, coste_anual: float,
             anios: int, probabilidad: float) -> pd.DataFrame:
    """Métricas de rentabilidad a partir de la rejilla de percentiles, sin
    volver a simular (precisión de 0,5 puntos en las probabilidades)."""
    valor_actual = np.asarray(valor_actual, dtype=float)
    precio = valor_actual * (1 + prima)
    descuento = (1 + coste_anual) ** anios
    col = int(np.argmin(np.abs(PERCENTILES - 100 * (1 - probabilidad))))
    mediana = rejilla[:, int(np.argmin(np.abs(PERCENTILES - 50)))]
    oferta = rejilla[:, col] / descuento
    return pd.DataFrame({
        "precio": precio,
        "oferta_maxima": oferta,
        "margen": oferta / precio - 1,
        "rentabilidad_mediana": mediana / descuento / precio - 1,
        "beneficio_esperado": media / descuento - precio,
        "beneficio_mediano": mediana / descuento - precio,
        "beneficio_asegurado": oferta - precio,
        "prob_beneficio": (rejilla > (precio * descuento)[:, None]).mean(axis=1),
        "valor_mediano_venta": mediana,
    })
