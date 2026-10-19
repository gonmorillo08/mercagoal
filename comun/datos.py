"""
Carga de la tabla limpia de valores de mercado.

La tabla la generan los administradores en el proyecto TFG (pestaña
"Compilación") y se copia a mano a `datos/`. Esta aplicación solo la LEE: no
la modifica ni la regenera. Para actualizarla, basta con sustituir el
fichero de `datos/` por uno nuevo; la caché se invalida sola porque depende
de la fecha de modificación del fichero.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

DIR_DATOS = Path(__file__).resolve().parent.parent / "datos"
ARCHIVO_PARQUET = DIR_DATOS / "tabla_valores_mercado_limpia.parquet"
ARCHIVO_CSV = DIR_DATOS / "tabla_valores_mercado_limpia.csv"

COLUMNAS_TEXTO = ["nombre", "equipo", "liga", "Bandera", "posicion", "nacionalidad", "pie"]
TRADUCCION_PIE = {"left": "Izquierdo", "right": "Derecho", "both": "Ambos"}


def _ruta_disponible() -> Path | None:
    for ruta in (ARCHIVO_PARQUET, ARCHIVO_CSV):
        if ruta.exists():
            return ruta
    return None


@st.cache_data(show_spinner="Cargando la tabla de valores de mercado...")
def _leer(ruta: str, marca_tiempo: float) -> pd.DataFrame:
    """`marca_tiempo` solo sirve para invalidar la caché cuando se sustituye
    el fichero de datos."""
    ruta = Path(ruta)
    df = pd.read_parquet(ruta) if ruta.suffix == ".parquet" else pd.read_csv(ruta)
    df["fecha"] = pd.to_datetime(df["fecha"])
    for columna in COLUMNAS_TEXTO:
        if columna in df.columns:
            df[columna] = df[columna].astype("string")
    if "pie" in df.columns:
        df["pie"] = df["pie"].replace(TRADUCCION_PIE)
    return df.sort_values(["id_player", "fecha"]).reset_index(drop=True)


def cargar_tabla() -> pd.DataFrame:
    """Tabla limpia (parquet si existe, si no el CSV). Detiene la app con un
    mensaje claro si no hay ningún fichero de datos."""
    ruta = _ruta_disponible()
    if ruta is None:
        st.error(
            f"No se encontró la tabla limpia en `{DIR_DATOS}`. Copia en esa carpeta "
            "`tabla_valores_mercado_limpia.parquet` (o `.csv`) generada desde la app de "
            "administración."
        )
        st.stop()
    return _leer(str(ruta), ruta.stat().st_mtime)


def fecha_actualizacion() -> datetime | None:
    ruta = _ruta_disponible()
    return datetime.fromtimestamp(ruta.stat().st_mtime) if ruta else None


@st.cache_data
def etiquetas_jugadores(_perfiles: pd.DataFrame, marca: float, con_valor: bool = False) -> dict[int, str]:
    """Etiqueta legible de cada jugador ("nombre · nacionalidad · equipo
    actual", y opcionalmente su valor) para los desplegables de selección.
    Se calcula una sola vez y de forma vectorizada: con ~41.000 jugadores,
    construirla opción a opción dentro del `format_func` del desplegable
    tardaba más de un minuto en cada carga de la página."""
    etiqueta = (
        _perfiles["nombre"].astype(str) + " · " + _perfiles["nacionalidad"].astype(str)
        + " · " + _perfiles["equipo_actual"].astype(str)
    )
    if con_valor:
        from comun.estilo import euros
        etiqueta = etiqueta + " (" + _perfiles["valor_actual"].map(euros) + ")"
    return dict(zip(_perfiles.index.tolist(), etiqueta.tolist()))


@st.cache_data
def perfil_jugadores(_df: pd.DataFrame, marca: float) -> pd.DataFrame:
    """Una fila por jugador: datos estáticos, último valor y equipo, pico
    de valor y periodo cubierto. Base de la ficha de jugador y de los
    rankings."""
    ordenado = _df.sort_values("fecha")
    ultimo = ordenado.groupby("id_player").tail(1).set_index("id_player")
    pico_idx = _df.groupby("id_player")["valor_mercado_eur"].idxmax()
    pico = _df.loc[pico_idx].set_index("id_player")
    agregado = _df.groupby("id_player").agg(
        primera_fecha=("fecha", "min"), ultima_fecha=("fecha", "max"),
        n_registros=("fecha", "size"), n_equipos=("equipo", "nunique"),
    )
    return pd.DataFrame(
        {
            "nombre": ultimo["nombre"],
            "nacionalidad": ultimo["nacionalidad"],
            "posicion": ultimo["posicion"],
            "equipo_actual": ultimo["equipo"],
            "liga_actual": ultimo["liga"],
            "bandera_liga": ultimo["Bandera"],
            "valor_actual": ultimo["valor_mercado_eur"],
            "valor_pico": pico["valor_mercado_eur"],
            "fecha_pico": pico["fecha"],
            "equipo_pico": pico["equipo"],
            # Columnas del jugador que solo existen en versiones de la tabla
            # limpia que ya incluyen los datos de players.csv.
            "edad_actual": ultimo["edad"] if "edad" in ultimo else pd.NA,
            "edad_pico": pico["edad"] if "edad" in pico else pd.NA,
            "altura_cm": ultimo["altura_cm"] if "altura_cm" in ultimo else pd.NA,
            "pie": ultimo["pie"] if "pie" in ultimo else pd.NA,
        }
    ).join(agregado)
