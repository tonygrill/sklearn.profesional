# ML Model Selection and Prediction API

## Introducción

Este proyecto tiene como objetivo automatizar la selección del mejor modelo de Machine Learning (ML) para predecir el coeficiente de felicidad de los países. Utilizamos un conjunto de datos que contiene varios indicadores de felicidad de diferentes países. El proyecto incluye una API construida con Flask para realizar predicciones basadas en el modelo entrenado.

## Estructura del Proyecto

El proyecto está organizado en los siguientes archivos:

- `main.py`: Script principal que carga los datos, prepara las características y el objetivo, y entrena los modelos.
- `utils.py`: Contiene utilidades para cargar datos, dividir características y objetivos, y exportar el mejor modelo.
- `models.py`: Define los modelos y los hiperparámetros, y realiza la búsqueda en cuadrícula para encontrar el mejor modelo.
- `server.py`: Implementa una API Flask para predecir la felicidad de un país dado un conjunto de características.

## Datos

El dataset utilizado, `felicidad.csv`, contiene los siguientes indicadores de felicidad de diferentes países:

- `score`: Puntuación de felicidad.
- `rank`: Rango de felicidad.
- `country`: Nombre del país.
- Otros indicadores relevantes que contribuyen a la puntuación de felicidad.

## Instalación

1. Clona el repositorio:
    ```sh
    git clone <URL del repositorio>
    cd <nombre del repositorio>
    ```

2. Crea y activa un entorno virtual:
    - En macOS/Linux:
      ```sh
      python3 -m venv venv
      source venv/bin/activate
      ```
    - En Windows:
      ```sh
      python -m venv venv
      venv\Scripts\activate
      ```

3. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Generación del `requirements.txt`

Para generar el archivo `requirements.txt` con todas las dependencias necesarias, utiliza el siguiente comando:
```sh
pip freeze > requirements.txt
