A continuación te propongo un ejemplo de configuración para el archivo setup.py y un README.md para el repositorio de utils_functions.

---

### Ejemplo de install_requires (setup.py)

```python
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

setup(
    name="utils_functions",
    version="0.1",
    packages=find_packages(),
    description="Colección de funciones utilitarias para facilitar operaciones en la nube y el procesamiento de datos (GCP, PDFs, tablas y más).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DavidP0011/utils_functions",
    author="DavidP0011",
    author_email="tu_email@ejemplo.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        "google-cloud-secretmanager>=2.7.0",  # Para funciones de inicialización GCP (dpm_GCP_ini_utils.py)
        "google-cloud-storage>=2.8.0",        # Para manejo de GCS
        "google-cloud-bigquery>=2.34.0",        # Si se requiere en operaciones GCP
        "PyPDF2>=3.0.0",                      # Para procesamiento de PDFs (dpm_pdf_utils.py)
        "pandas>=1.3.0",                      # Para manipulación de datos en dpm_tables.py y otros módulos
        "tabulate>=0.8.10",                   # Opcional: para formateo de tablas en dpm_tables.py
        "requests>=2.25.0",                   # Para descargas y scraping en algunos módulos
        "numpy>=1.18.0"                       # Soporte numérico en funciones de tablas
    ],
    python_requires=">=3.7",
)
```

En este ejemplo se incluyen las dependencias principales que usan los módulos:

- **dpm_GCP_ini_utils.py:** Probablemente utiliza las librerías de GCP, por ello se incluyen google-cloud-secretmanager, google-cloud-storage y google-cloud-bigquery.
- **dpm_pdf_utils.py:** Se basa en PyPDF2 para procesar archivos PDF.
- **dpm_tables.py:** Usa pandas (y opcionalmente tabulate y numpy) para la manipulación de datos tabulares.
- **dpm_old.py:** Al ser funciones legacy, se asume que utilizan únicamente módulos estándar o los mismos paquetes que ya se requieren.

---

### Ejemplo de README.md

```markdown
# utils_functions

**utils_functions** es un paquete Python que agrupa una serie de funciones utilitarias para facilitar tareas comunes en el desarrollo de soluciones de procesamiento de datos y operaciones en la nube. Este repositorio incluye módulos para:

- **Inicialización y autenticación en GCP:**  
  Funciones en `dpm_GCP_ini_utils.py` que simplifican la configuración y autenticación para acceder a servicios de Google Cloud Platform.

- **Funciones legacy:**  
  El módulo `dpm_old.py` contiene funciones antiguas o heredadas que pueden seguir siendo útiles en ciertos contextos.

- **Procesamiento de PDFs:**  
  El módulo `dpm_pdf_utils.py` ofrece utilidades para extraer información y manipular documentos PDF, aprovechando la librería PyPDF2.

- **Manejo de tablas y datos:**  
  En `dpm_tables.py` se incluyen funciones para procesar, transformar y formatear datos tabulares utilizando pandas y herramientas complementarias.

## Características

- **Integración con GCP:**  
  Facilita la autenticación y conexión a servicios de Google Cloud (Secret Manager, Storage y BigQuery).

- **Procesamiento de documentos PDF:**  
  Extrae contenido de archivos PDF para su posterior análisis o transformación.

- **Utilidades para manejo de tablas:**  
  Provee funciones para la manipulación de DataFrames y formateo de tablas.

- **Funciones Legacy:**  
  Incluye funciones antiguas que aún pueden resultar útiles en ciertos proyectos.

## Instalación

Puedes instalar el paquete directamente desde GitHub:

```bash
pip install -q git+https://github.com/DavidP0011/utils_functions.git
```

O clonar el repositorio y luego instalarlo:

```bash
git clone https://github.com/DavidP0011/utils_functions.git
cd utils_functions
pip install .
```

## Requisitos

El paquete depende de las siguientes librerías, que se instalarán automáticamente:

- **google-cloud-secretmanager** (>=2.7.0)
- **google-cloud-storage** (>=2.8.0)
- **google-cloud-bigquery** (>=2.34.0)
- **PyPDF2** (>=3.0.0)
- **pandas** (>=1.3.0)
- **tabulate** (>=0.8.10)
- **requests** (>=2.25.0)
- **numpy** (>=1.18.0)

## Uso

El paquete está organizado en módulos dentro del directorio `dpm_functions`. Aquí se explica brevemente qué ofrece cada uno:

- **dpm_GCP_ini_utils.py:**  
  Funciones para facilitar la inicialización y autenticación en GCP, permitiendo usar claves locales o cargarlas desde Secret Manager.

- **dpm_old.py:**  
  Funciones legacy para tareas que, aunque antiguas, siguen siendo útiles.

- **dpm_pdf_utils.py:**  
  Utilidades para la lectura y extracción de información de archivos PDF.

- **dpm_tables.py:**  
  Funciones para el procesamiento y formateo de datos tabulares con pandas y, opcionalmente, tabulate.

### Ejemplo de uso

A modo de ejemplo, para inicializar la autenticación en GCP utilizando las utilidades, podrías hacer lo siguiente:

```python
from dpm_functions import dpm_GCP_ini_utils

config = {
    "ini_environment_identificated": "COLAB",
    "json_keyfile_colab": "/ruta/a/tu/credencial.json"
}

# Supongamos que dpm_GCP_ini_utils expone una función para cargar credenciales:
credentials = dpm_GCP_ini_utils.get_credentials(config, "mi-proyecto")
print("Credenciales cargadas:", credentials)
```

Consulta la documentación interna de cada módulo para más detalles sobre la configuración y los parámetros requeridos.

## Contribuir

Las contribuciones son bienvenidas. Si deseas mejorar el paquete:

1. Haz un fork del repositorio.
2. Crea una rama con tus cambios:  
   `git checkout -b feature/nueva-funcionalidad`
3. Realiza tus cambios y haz commit:  
   `git commit -am "Agrega nueva funcionalidad"`
4. Sube tus cambios a tu fork:  
   `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request en este repositorio.

## Licencia

Este proyecto se distribuye bajo la [Licencia MIT](LICENSE).

## Contacto

Para consultas, sugerencias o reportar errores, abre un issue en GitHub o contacta al autor.
```

---

Este ejemplo de setup.py e README.md está basado en los módulos encontrados en el repositorio y en las dependencias requeridas para que todas las funciones operen correctamente. ¿Te parece adecuado o deseas algún ajuste adicional?
