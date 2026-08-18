# Detección de fraude — Data Scientist Technical Challenge

Solución al reto de prevención de fraude en transacciones de pago. El objetivo
es clasificar transacciones como fraudulentas o no, además de maximizar
la ganancia de la empresa dada la economía del problema:

- Aprobar una transacción legítima genera una ganancia del **25% del monto**.
- Aprobar una transacción fraudulenta pierde el **100% del monto**.
- Rechazar una transacción no genera ganancia ni pérdida.

Por eso la métrica central del proyecto es la **ganancia neta simulada**. El umbral de decisión se elige para maximizar esa ganancia
sobre datos de validación, usando probabilidades calibradas.

## Estructura del repositorio

```
fraud-detection/
├── data/                    # dataset (no versionado; ver "Datos" abajo)
├── notebooks/
│   ├── 01_eda.ipynb                        # análisis exploratorio
│   ├── 02_preprocessing_features.ipynb     # limpieza, split temporal, features
│   ├── 03_baseline_y_modelos.ipynb         # baselines, logística, LGBM + Optuna
│   └── 04_calibracion_umbral_evaluacion.ipynb  # calibración, umbral, evaluación
├── src/
│   ├── data.py              # carga y limpieza
│   ├── features.py          # ingeniería de features
│   ├── profit.py            # función de ganancia (métrica central)
│   └── model.py             # entrenamiento y calibración
├── tests/
│   └── test_profit.py       # tests de la función de ganancia
└── reports/
    ├── figures/             # figuras generadas
    └── informe.pdf          # informe final
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datos


## Ejecución

Correr los notebooks en orden (01 → 04). Cada uno importa las funciones desde
`src/`. Para reproducibilidad, todos usan una semilla fija.

## Tests

```bash
pytest
```

Los tests verifican la función de ganancia, que es el corazón de la solución.

## Reproducibilidad

- Semilla fija en todos los notebooks y en el entrenamiento.
- Versiones de dependencias fijadas en `requirements.txt`.
- Split **temporal** (out-of-time) para que la evaluación en laboratorio se
  aproxime al comportamiento en producción.