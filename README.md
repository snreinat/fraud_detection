# Detección de fraude

Solución al challenge técnico de prevención de fraude en transacciones de pago.

El objetivo del proyecto no es únicamente clasificar transacciones como fraudulentas o legítimas, sino construir una política de decisión que permita **maximizar la ganancia simulada de la empresa** bajo las condiciones económicas definidas en el reto.

## Objetivo de negocio

La función de ganancia se construye a partir de las reglas del challenge:

- Aprobar una transacción **legítima** genera una ganancia equivalente al **25% del monto**.
- Aprobar una transacción **fraudulenta** genera una pérdida equivalente al **100% del monto**.
- Dado que el enunciado no especifica un costo asociado al rechazo de una transacción, se asume una ganancia/pérdida igual a **0** para las transacciones rechazadas.

Por tanto, la métrica central del proyecto es la **ganancia neta simulada**, mientras que métricas como PR-AUC, precision, recall y F1 se utilizan para evaluar la calidad predictiva del modelo.

## Resultado principal

Evaluación sobre el conjunto de test **out-of-time**, utilizando el umbral seleccionado previamente sobre el conjunto de validación:

| Estrategia | Ganancia simulada en test |
|---|---:|
| Rechazar todo | $0 |
| Aprobar todo | $138.456 |
| Baseline basado en `score` | $159.514 |
| **XGBoost + Platt scaling + umbral optimizado** | **$173.489** |
| Máximo posible (oráculo) | $217.836 |

Bajo este escenario, el modelo incrementa la ganancia simulada aproximadamente en:

- **$35.033 frente a aprobar todas las transacciones**.
- **$13.975 frente al baseline basado en la variable `score`**.

Como referencia adicional, la ganancia obtenida por el modelo representa aproximadamente el *79,6%* de la ganancia máxima teórica, definida por un oráculo que conoce de antemano la etiqueta real de cada transacción.
 
## Estructura del repositorio

```text
fraud_detection/
├── data/
│   └── dataset.csv #No se deja en el repositorio
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_features.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_calibration_threshold_evaluation.ipynb
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── profit.py
│   └── utils.py
│
├── tests/
│   └── test_profit.py
│
├── requirements.txt
└── README.md
```
 
### Descripción
- data.py: carga y limpieza inicial del dataset.
- features.py: split temporal y construcción del preprocesamiento.
- model.py: definición de pipelines y modelos.
- profit.py: implementación de la función de ganancia.
- utils.py: funciones auxiliares de visualización.
- tests/test_profit.py: pruebas unitarias de la métrica de negocio.


## Instalación
 
Se recomienda utilizar  **Python 3.12**.
 
Crear un entorno virtual: 

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
 
## Datos
 
El dataset **no se versiona** dentro del repositorio. El archivo suministrado para el challenge debe ubicarse en:
 
```
data/dataset.csv
```
 
La carga se realiza de manera centralizada mediante `src/data.py`. El dataset contiene
150.000 transacciones y 19 columnas, incluyendo: 

- variables anonimizadas;
- `fecha`;
- `monto`;
- `score`;
- `fraude` (1 = transacción fraudulenta, 0 = transacción legítima).

La variable `score` se conserva como una señal disponible en el dataset y también se utiliza como baseline de comparación. Su interpretación exacta como sistema antifraude preexistente no se asume más allá de la información proporcionada en los datos.
 

## Ejecución

Los notebooks son autocontenidos y siguen una secuencia lógica de análisis:

```text
01 → 02 → 03 → 04
```

Para comprobar la reproducibilidad, cada notebook debe ejecutarse desde un kernel limpio:

```text
Restart Kernel → Run All
```

### 1. `01_eda.ipynb`

Análisis exploratorio del dataset:

- distribución de la variable objetivo;
- valores faltantes;
- variables numéricas y categóricas;
- comportamiento temporal;
- análisis de posibles variables sin señal o con riesgo de leakage.

### 2. `02_preprocessing_features.ipynb`

Preparación de los datos:

- partición temporal train / validation / test;
- limpieza;
- imputación;
- codificación de variables categóricas;
- construcción del pipeline de preprocesamiento.

Todos los parámetros del preprocesamiento se ajustan exclusivamente con datos de entrenamiento.

### 3. `03_model_comparison.ipynb`

Entrenamiento y comparación de modelos:

- regresión logística como baseline;
- XGBoost;
- manejo del desbalance;
- optimización de hiperparámetros mediante Optuna;
- evaluación mediante PR-AUC.

### 4. `04_calibration_threshold_evaluation.ipynb`

Evaluación orientada a negocio:

- calibración de scores;
- selección del umbral de decisión;
- maximización de la ganancia simulada;
- comparación contra baselines;
- evaluación final sobre el conjunto out-of-time.

---

## Metodología

### 1. Análisis exploratorio

Se realiza un análisis univariado y bivariado para comprender:

- distribución de las variables;
- prevalencia del fraude;
- faltantes;
- cardinalidad;
- relaciones con la variable objetivo;
- posibles problemas de calidad o fuga de información.

---

### 2. Validación temporal

Las transacciones se ordenan cronológicamente y se dividen en:

```text
TRAIN → VALIDATION → TEST OOT
```

El objetivo es aproximar el escenario real de producción:

> entrenar con información histórica y evaluar sobre transacciones posteriores.

El conjunto de test se mantiene fuera del proceso de selección de hiperparámetros y umbrales.

---

### 3. Preprocesamiento

Las transformaciones se ajustan exclusivamente utilizando el conjunto de entrenamiento y posteriormente se aplican a validación y test.

Se utilizan, según el tipo de variable:

- imputación;
- One-Hot Encoding para categóricas de baja cardinalidad;
- Target Encoding para variables de mayor cardinalidad.

Esto evita que información futura influya en la construcción de las variables utilizadas por el modelo.

---

### 4. Modelado

Se comparan distintas familias de modelos:

- **Regresión logística**, como baseline lineal interpretable.
- **Random Forest**.
- **XGBoost**.
- **LightGBM**.
- **SVM con kernel RBF**.
- **MLP (red neuronal)**.

Los mejores resultados de ranking se obtienen con los modelos de gradient boosting. 
XGBoost se selecciona como modelo final por su desempeño, estabilidad y su integración con las etapas posteriores de calibración, optimización del umbral e interpretabilidad.

El desbalance de clases se aborda mediante ponderación durante el entrenamiento, evitando modificar artificialmente la distribución observada mediante oversampling sintético.

Los hiperparámetros de XGBoost se optimizan mediante **Optuna**.

---

### 5. Evaluación del ranking

Debido al desbalance de clases, la métrica principal para evaluar la capacidad de ranking es:

```text
PR-AUC
```

También se reportan:

- Precision;
- Recall;
- F1;
- falsos positivos;
- falsos negativos.

Estas métricas permiten evaluar la capacidad predictiva, pero no determinan por sí solas la política óptima de aprobación o rechazo.

---

### 6. Calibración

Se aplica calibración mediante **Platt scaling** para mejorar la interpretación probabilística de los scores producidos por el modelo.

La calibración permite comparar de forma más razonable el umbral seleccionado empíricamente con el umbral económico derivado de la función de ganancia.

---

### 7. Optimización del umbral

Una vez entrenado el modelo, la decisión final se separa del problema de ranking.

Para cada transacción, el sistema debe decidir entre:

```text
aprobar
rechazar
```

La ganancia simulada asociada a una decisión se define como:

```text
legítima aprobada      → +0.25 × monto
fraude aprobado        → -1.00 × monto
transacción rechazada  → 0
```

El umbral operativo se selecciona sobre el conjunto de validación maximizando la ganancia total.

Bajo probabilidades perfectamente calibradas y los supuestos económicos anteriores, el punto de equilibrio teórico se encuentra alrededor de una probabilidad de fraude de:

```text
0.20
```

El umbral empírico obtenido en validación se compara con este valor como análisis de consistencia económica.

---

## Tests

La función de ganancia se encuentra separada en:

```text
src/profit.py
```

y cuenta con pruebas unitarias:

```bash
pytest
```

Estas pruebas verifican la correcta asignación de ganancias y pérdidas para las distintas combinaciones de:

- fraude / legítima;
- aprobada / rechazada.

Dado que la función de ganancia constituye la métrica principal de negocio, se mantiene aislada y testeada independientemente del modelo.

---

## Reproducibilidad

Para favorecer la reproducibilidad:

- se utiliza `RANDOM_STATE = 42`;
- las dependencias se fijan en `requirements.txt`;
- las transformaciones se encapsulan en funciones dentro de `src/`;
- el preprocesamiento se ajusta únicamente sobre datos de entrenamiento;
- se utiliza una partición temporal out-of-time;
- los notebooks pueden ejecutarse completamente desde un kernel limpio.

Con las versiones de dependencias y semillas especificadas, la ejecución completa debería producir resultados equivalentes a los reportados.

---

