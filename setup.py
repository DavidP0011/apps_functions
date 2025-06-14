from setuptools import setup, find_packages 
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (
    (here / "README.md").read_text(encoding="utf-8")
    if (here / "README.md").exists()
    else ""
)

setup(
    name='dpm_apps_functions',
    version='0.1',
    description='Conjunto de funciones para aplicaciones varias',
    url='https://github.com/DavidP0011/apps_functions',
    author='David Plaza', 
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    python_requires='>=3.7, <4',
    install_requires=[
        "pandas>=1.3.0",                    
        "gspread>=5.6.0",                   
        "oauth2client>=4.1.3",              
        "openai-whisper>=20230314",         
        "torch>=1.10.0",                    
        # Aquí viene la dependencia directa del repo common_functions
        "dpm_common_functions @ git+https://github.com/DavidP0011/common_functions.git@main#egg=dpm_common_functions",
    ],
    entry_points={
        # Si tuvieras scripts de consola, los listarías acá
    },
)
