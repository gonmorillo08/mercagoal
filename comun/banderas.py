"""
Bandera (imagen PNG de flagcdn.com) de cada país/nacionalidad.

`PAIS_A_CODIGO` es una copia del catálogo de `limpieza/nacionalidad_y_liga.py`
del proyecto de administración (TFG): país tal y como aparece en
`nacionalidad` y en `liga` de la tabla limpia -> código de flagcdn.com.
"""

from __future__ import annotations

import pandas as pd

_URL_FLAGCDN = "https://flagcdn.com/40x30/{codigo}.png"

PAIS_A_CODIGO: dict[str, str | None] = {
    "Afghanistan": "AF", "Albania": "AL", "Algeria": "DZ", "Andorra": "AD",
    "Angola": "AO", "Antigua and Barbuda": "AG", "Argentina": "AR", "Armenia": "AM",
    "Aruba": "AW", "Australia": "AU", "Austria": "AT", "Azerbaijan": "AZ",
    "Bahrain": "BH", "Bangladesh": "BD", "Barbados": "BB", "Belarus": "BY",
    "Belgium": "BE", "Benin": "BJ", "Bermuda": "BM", "Bolivia": "BO",
    "Bonaire": "BQ", "Bosnia-Herzegovina": "BA", "Botswana": "BW", "Brazil": "BR",
    "Brunei Darussalam": "BN", "Bulgaria": "BG", "Burkina Faso": "BF", "Burundi": "BI",
    "Cambodia": "KH", "Cameroon": "CM", "Canada": "CA", "Cape Verde": "CV",
    "Central African Republic": "CF", "Chad": "TD", "Chile": "CL", "China": "CN",
    "Chinese Taipei": "TW", "Colombia": "CO", "Comoros": "KM", "Congo": "CG",
    "Costa Rica": "CR", "Cote d'Ivoire": "CI", "Croatia": "HR", "Cuba": "CU",
    "Curacao": "CW", "Cyprus": "CY", "Czech Republic": "CZ", "DR Congo": "CD",
    "Denmark": "DK", "Dominican Republic": "DO", "Ecuador": "EC", "Egypt": "EG",
    "El Salvador": "SV", "England": "gb-eng", "Equatorial Guinea": "GQ", "Eritrea": "ER",
    "Estonia": "EE", "Ethiopia": "ET", "Faroe Islands": "FO", "Fiji": "FJ",
    "Finland": "FI", "France": "FR", "French Guiana": "GF", "Gabon": "GA",
    "Georgia": "GE", "Germany": "DE", "Ghana": "GH", "Gibraltar": "GI",
    "Greece": "GR", "Grenada": "GD", "Guadeloupe": "GP", "Guatemala": "GT",
    "Guinea": "GN", "Guinea-Bissau": "GW", "Guyana": "GY", "Haiti": "HT",
    "Honduras": "HN", "Hongkong": "HK", "Hungary": "HU", "Iceland": "IS",
    "India": "IN", "Indonesia": "ID", "Iran": "IR", "Iraq": "IQ",
    "Ireland": "IE", "Israel": "IL", "Italy": "IT", "Jamaica": "JM",
    "Japan": "JP", "Jordan": "JO", "Kazakhstan": "KZ", "Kenya": "KE",
    "Korea, North": "KP", "Korea, South": "KR", "Kosovo": "XK", "Kyrgyzstan": "KG",
    "Laos": "LA", "Latvia": "LV", "Lebanon": "LB", "Liberia": "LR",
    "Libya": "LY", "Liechtenstein": "LI", "Lithuania": "LT", "Luxembourg": "LU",
    "Macao": "MO", "Madagascar": "MG", "Malawi": "MW", "Malaysia": "MY",
    "Mali": "ML", "Malta": "MT", "Martinique": "MQ", "Mauritania": "MR",
    "Mauritius": "MU", "Mexico": "MX", "Moldova": "MD", "Monaco": "MC",
    "Montenegro": "ME", "Montserrat": "MS", "Morocco": "MA", "Mozambique": "MZ",
    "Myanmar": "MM", "Netherlands": "NL", "Neukaledonien": "NC", "New Caledonia": "NC",
    "New Zealand": "NZ", "Nicaragua": "NI", "Niger": "NE", "Nigeria": "NG",
    "North Macedonia": "MK", "Northern Ireland": "gb-nir", "Norway": "NO", "Oman": "OM",
    "Pakistan": "PK", "Palestine": "PS", "Panama": "PA", "Papua New Guinea": "PG",
    "Paraguay": "PY", "Peru": "PE", "Philippines": "PH", "Poland": "PL",
    "Portugal": "PT", "Puerto Rico": "PR", "Qatar": "QA", "Romania": "RO",
    "Russia": "RU", "Rwanda": "RW", "Réunion": "RE", "Saint-Martin": "MF",
    "San Marino": "SM", "Sao Tome and Principe": "ST", "Saudi Arabia": "SA", "Scotland": "gb-sct",
    "Senegal": "SN", "Serbia": "RS", "Seychelles": "SC", "Sierra Leone": "SL",
    "Singapore": "SG", "Sint Maarten": "SX", "Slovakia": "SK", "Slovenia": "SI",
    "Somalia": "SO", "South Africa": "ZA", "Southern Sudan": "SS", "Spain": "ES",
    "St. Kitts & Nevis": "KN", "St. Lucia": "LC", "Sudan": "SD", "Suriname": "SR",
    "Sweden": "SE", "Switzerland": "CH", "Syria": "SY", "Tahiti": "PF",
    "Tajikistan": "TJ", "Tanzania": "TZ", "Thailand": "TH", "The Gambia": "GM",
    "Togo": "TG", "Trinidad and Tobago": "TT", "Tunisia": "TN", "Turkey": "TR",
    "Turkmenistan": "TM", "Türkiye": "TR", "Uganda": "UG", "Ukraine": "UA",
    "United Arab Emirates": "AE", "United States": "US", "Uruguay": "UY", "Uzbekistan": "UZ",
    "Vanuatu": "VU", "Venezuela": "VE", "Vietnam": "VN", "Wales": "gb-wls",
    "Yemen": "YE", "Zambia": "ZM", "Zimbabwe": "ZW",
}


def url_bandera(pais) -> str:
    """URL de la bandera de `pais` o cadena vacía si no está catalogado."""
    if pais is None or pd.isna(pais):
        return ""
    codigo = PAIS_A_CODIGO.get(str(pais))
    return _URL_FLAGCDN.format(codigo=codigo.lower()) if codigo else ""


def banderas_en_eje_y(fig, separacion_px: int = 34) -> None:
    """Añade la bandera de cada categoría del eje Y de un gráfico de barras
    HORIZONTAL (una imagen por barra, justo a la izquierda del eje) y separa
    los nombres para que la bandera quede entre el nombre y la barra. Las
    categorías sin bandera catalogada simplemente se quedan sin imagen."""
    if not fig.data:
        return
    categorias = list(dict.fromkeys(fig.data[0].y))
    for posicion, categoria in enumerate(categorias):
        url = url_bandera(categoria)
        if not url:
            continue
        fig.add_layout_image(dict(
            source=url, xref="paper", yref="y", x=-0.006, y=posicion,
            sizex=0.045, sizey=0.62, xanchor="right", yanchor="middle", sizing="contain", layer="above",
        ))
    fig.update_yaxes(ticklabelstandoff=separacion_px)
