"""
Registro de las páginas de la aplicación, para que un módulo pueda enlazar a
otro (p. ej. `st.switch_page(PAGINAS["fichajes"])`) sin importar `app.py`.
`app.py` lo rellena al crear la navegación.
"""

PAGINAS: dict = {}
