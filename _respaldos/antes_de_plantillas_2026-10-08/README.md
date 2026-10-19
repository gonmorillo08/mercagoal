# Portal de usuario — Valores de mercado

Aplicación de **consulta** para usuarios/clientes. Trabaja sobre la tabla
limpia de valores de mercado que generan los administradores en el proyecto
`TFG` (pestaña *Limpieza de datos → Compilación*).

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

```
TFG_USUARIO/
├── app.py                  Aplicación: página de inicio y navegación entre módulos
├── .streamlit/config.toml  Tema visual
├── datos/                  Tabla limpia (parquet y csv), copiada a mano desde TFG
├── comun/                  Carga de datos (cacheada), estilo y catálogo de banderas
├── exploracion/            Módulo "Explorador de datos"
├── visualizacion/          Módulo "Visualización de datos" (gráficas y red de traspasos)
├── prediccion/             Módulo "Predicción de valores de mercado" (Monte Carlo por posición, edad y etapa)
└── fichajes/               Módulo "Medidor de fichajes" (oferta máxima para obtener beneficio con probabilidad x)
```

## Actualizar los datos

1. En el proyecto `TFG`, compilar la tabla limpia (pestaña *Compilación*).
2. Copiar `TFG/ingest/datos/tabla_valores_mercado_limpia.parquet` (y el `.csv`)
   a `TFG_USUARIO/datos/`, sustituyendo los anteriores.

La aplicación detecta el cambio sola (la caché depende de la fecha del fichero).

## Añadir un módulo

Crear una carpeta con un `__init__.py` que exponga una función `render()` y
registrarla en `PAGINAS` dentro de `app.py`.
