"""
Estilo visual común: CSS de la aplicación, plantilla de Plotly y pequeños
componentes reutilizables (cabecera de página, tarjetas de KPI, formato de
cifras en euros).
"""

from __future__ import annotations

import html
from pathlib import Path

import plotly.express as px
import plotly.io as pio
import streamlit as st

# Colores del logotipo de Mercagoal.
COLOR_OSCURO = "#10231C"
COLOR_PRIMARIO = "#1A7F4B"
COLOR_ACENTO = "#3DDC84"
PALETA = ["#1A7F4B", "#F59E0B", "#3B82F6", "#3DDC84", "#EF4444", "#8B5CF6", "#4B5E55", "#EC4899",
          "#14B8A6", "#F97316"]
DIR_ASSETS = Path(__file__).resolve().parent.parent / "assets"

_CSS = f"""
<style>
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; color: {COLOR_OSCURO}; }}

.cabecera {{
    background: linear-gradient(120deg, {COLOR_OSCURO} 0%, {COLOR_OSCURO} 55%, {COLOR_PRIMARIO} 100%);
    color: #fff; border-radius: 14px; padding: 1.4rem 1.8rem; margin-bottom: 1.4rem;
    border-bottom: 4px solid {COLOR_ACENTO};
    box-shadow: 0 6px 18px rgba(16, 35, 28, 0.18);
}}
.cabecera h1 {{ color: #fff; font-size: 1.9rem; margin: 0; padding: 0; }}
.cabecera p {{ color: rgba(255,255,255,0.88); margin: 0.35rem 0 0 0; font-size: 0.98rem; }}

.kpi {{
    background: #F6F8F6;
    border: 1px solid rgba(16, 35, 28, 0.10); border-left: 4px solid {COLOR_PRIMARIO};
    border-radius: 10px; padding: 0.85rem 1rem; height: 100%;
}}
.kpi .etiqueta {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.7; }}
.kpi .valor {{ font-size: 1.55rem; font-weight: 700; line-height: 1.25; margin-top: 0.15rem; }}
.kpi .detalle {{ font-size: 0.82rem; opacity: 0.75; margin-top: 0.1rem; white-space: nowrap;
                overflow: hidden; text-overflow: ellipsis; }}

.ficha {{
    border: 1px solid rgba(16, 35, 28, 0.12); border-radius: 12px; padding: 1.1rem 1.3rem;
    background: #F6F8F6;
}}
.ficha h3 {{ margin: 0 0 0.3rem 0; }}
.ficha .dato {{ margin: 0.15rem 0; font-size: 0.95rem; }}
.ficha img {{ vertical-align: middle; border-radius: 2px; margin-right: 0.35rem; }}

div[data-testid="stTabs"] button[role="tab"] p {{ font-size: 0.98rem; font-weight: 600; }}
</style>
"""


def aplicar_estilo() -> None:
    """CSS global y plantilla por defecto de Plotly (colores y tipografía
    coherentes en todas las gráficas)."""
    st.markdown(_CSS, unsafe_allow_html=True)
    plantilla = pio.templates["plotly_white"]
    plantilla.layout.colorway = PALETA
    plantilla.layout.font.family = "Outfit, Segoe UI, sans-serif"
    plantilla.layout.font.color = COLOR_OSCURO
    plantilla.layout.title.font.size = 17
    plantilla.layout.hoverlabel.bgcolor = "white"
    pio.templates["tfg"] = plantilla
    pio.templates.default = "tfg"
    px.defaults.template = "tfg"
    px.defaults.color_discrete_sequence = PALETA


def cabecera(titulo: str, subtitulo: str = "") -> None:
    st.markdown(
        f'<div class="cabecera"><h1>{html.escape(titulo)}</h1>'
        + (f"<p>{html.escape(subtitulo)}</p>" if subtitulo else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def euros(valor: float | None, compacto: bool = True) -> str:
    """12500000 -> '12,5 M€' (compacto) o '12.500.000 €'."""
    if valor is None or valor != valor:  # NaN
        return "—"
    if not compacto:
        return f"{valor:,.0f} €".replace(",", ".")
    for limite, sufijo in ((1e9, "mil M€"), (1e6, "M€"), (1e3, "k€")):
        if abs(valor) >= limite:
            return f"{valor / limite:.1f} {sufijo}".replace(".", ",")
    return f"{valor:.0f} €"


def numero(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def tarjetas_kpi(tarjetas: list[tuple[str, str, str]]) -> None:
    """Fila de tarjetas `(etiqueta, valor, detalle)`."""
    columnas = st.columns(len(tarjetas))
    for columna, (etiqueta, valor, detalle) in zip(columnas, tarjetas):
        columna.markdown(
            f'<div class="kpi"><div class="etiqueta">{html.escape(etiqueta)}</div>'
            f'<div class="valor">{html.escape(valor)}</div>'
            f'<div class="detalle" title="{html.escape(detalle)}">{html.escape(detalle) or "&nbsp;"}</div></div>',
            unsafe_allow_html=True,
        )
