from setuptools import setup, find_packages
import pathlib

# Directorio actual del archivo setup.py
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

setup(
    name='dpm_functions',
    version='0.1',
    description='Conjunto de funciones para aplicaciones varias',
    url='https://github.com/DavidP0011/etl_functions',
    author='David Plaza', 
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    python_requires='>=3.7, <4',
    install_requires=[
        "pandas>=1.3.0",                    # Para manipulación de DataFrames y operaciones con tablas
        "gspread>=5.6.0",                   # Para interactuar con Google Sheets
        "oauth2client>=4.1.3",              # Para autenticación con Google Sheets mediante ServiceAccountCredentials
        "openai-whisper>=20230314",         # Para la transcripción de audio/video usando Whisper
        "torch>=1.10.0"                     # Requerido para ejecutar el modelo Whisper (PyTorch)
    ]


    entry_points={
        # Scripts ejecutables desde la línea de comando
        # 'console_scripts': [
        #     'nombre_comando=dpm_functions.modulo:funcion_principal',
        # ],
    },
)
