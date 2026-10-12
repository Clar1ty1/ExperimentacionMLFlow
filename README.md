# California Housing — Random Forest con MLflow

## Autor: José Alberto Velázquez Trujillo A01369142

## Descripción

Proyecto de entrenamiento de un modelo de regresión `RandomForestRegressor` para predecir el valor medio de las viviendas del conjunto de datos California Housing.

El proyecto utiliza **scikit-learn** para el procesamiento y entrenamiento, y **MLflow Tracking** para registrar los hiperparámetros, las métricas de evaluación y el modelo entrenado.

## Requisitos

- Python 3.10 o superior.
- Dataset `housing.csv`.
- Dependencias incluidas en `requirements.txt`.

## Instalación

Clona el repositorio y entra en su directorio:

```bash
git clone https://github.com/TU-USUARIO/california-housing-mlflow.git
cd california-housing-mlflow
```

Crea y activa un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

En Windows, activa el entorno con:

```bash
.venv\Scripts\activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Coloca `housing.csv` en el directorio raíz del proyecto.

## Ejecución

Ejecuta el entrenamiento con los parámetros predeterminados:

```bash
python train.py
```

También puedes modificar los hiperparámetros desde la terminal:

```bash
python train.py --n-estimators 150 --max-depth 15 --min-samples-leaf 3 --seed 42
```

### Hiperparámetros disponibles

| Parámetro | Descripción | Valor predeterminado |
|---|---|---:|
| `--n-estimators` | Número de árboles del bosque | 100 |
| `--max-depth` | Profundidad máxima de cada árbol | 10 |
| `--min-samples-leaf` | Número mínimo de muestras por hoja | 2 |
| `--seed` | Semilla aleatoria para reproducibilidad | 42 |

## Métricas de evaluación

El modelo calcula las siguientes métricas para los conjuntos de validación y prueba:

- **MAE:** error absoluto medio.
- **RMSE:** raíz del error cuadrático medio.
- **R²:** proporción de variabilidad explicada por el modelo.

Las métricas se muestran en la terminal, se registran en MLflow y se guardan en el artefacto `metrics.json`.

## MLflow Tracking

Cada ejecución registra los hiperparámetros, las métricas y el modelo entrenado en el experimento `California Housing - Random Forest`.

Para consultar los experimentos, inicia la interfaz de MLflow:

```bash
mlflow ui
```

Abre en tu navegador:

http://127.0.0.1:5000

Ejecuta `train.py` varias veces con diferentes hiperparámetros para comparar los resultados dentro del mismo experimento.

## Estructura del proyecto

```text
california-housing-mlflow/
├── train.py
├── requirements.txt
├── housing.csv
├── metrics.json
├── .gitignore
└── README.md
```

`metrics.json` se genera durante la ejecución. Si se ejecuta el entrenamiento varias veces, su contenido se sobrescribe, mientras que MLflow conserva el registro de cada ejecución.

## Reproducibilidad

La semilla aleatoria se configura mediante `--seed`. Para reproducir los experimentos, utiliza la misma versión de Python, las dependencias especificadas en `requirements.txt`, el mismo conjunto de datos y los mismos hiperparámetros.

**Nota:** el modelo se registra utilizando el formato `pickle`. Carga únicamente modelos cuyo origen sea confiable.
