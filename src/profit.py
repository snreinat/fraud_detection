"""
Función de ganancia:

- Si se aprueba una transacción legítima (y=0):  +GAIN_RATE * monto   (ganancia 25%)
- Si se aprueba una transacción fraudulenta (y=1):  -LOSS_RATE * monto    (se pierde 100%)
- Rechazar (cualquiera):       0

Umbral de equilibrio: se rechaza cuando la ganancia esperada de aprobar es
negativa. Igualando a cero:
    GAIN_RATE*(1-p) - LOSS_RATE*p = 0  ->  p* = GAIN_RATE/(GAIN_RATE+LOSS_RATE)
Con 0.25 y 1.0 da p* = 0.20.
"""
import numpy as np

GAIN_RATE = 0.25   # ganancia sobre una legítima aprobada
LOSS_RATE = 1.0    # pérdida sobre un fraude aprobado
BREAK_EVEN_THRESHOLD = GAIN_RATE / (GAIN_RATE + LOSS_RATE)  # = 0.20


#Valor de APROBAR cada transacción: +25% del monto si es legítima, -100% del monto si es fraude.
def transaction_value(y_true, amount):
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    return np.where(y_true == 1, -LOSS_RATE * amount, GAIN_RATE * amount)

#Ganancia total dada una decisión binaria de aprobar/rechazar. approve_mask: booleano (True = aprobar). Las rechazadas aportan 0.
def total_profit(y_true, amount, approve_mask):
    approve_mask = np.asarray(approve_mask, dtype=bool)
    values = transaction_value(y_true, amount)
    return float(values[approve_mask].sum())

#Ganancia total aprobando cuando p_fraud <= threshold.
def profit_from_threshold(y_true, amount, p_fraud, threshold=BREAK_EVEN_THRESHOLD):
    p_fraud = np.asarray(p_fraud, dtype=float)
    return total_profit(y_true, amount, p_fraud <= threshold)


def find_best_threshold(y_true, amount, p_fraud, grid=None):
    """Busca el umbral que maximiza la ganancia sobre una grilla.
    Elegir SIEMPRE sobre validación, no sobre test.
    Devuelve (mejor_umbral, mejor_ganancia, grid, ganancias)."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    grid = np.asarray(grid, dtype=float)
    profits = np.array([profit_from_threshold(y_true, amount, p_fraud, t) for t in grid])
    i = int(np.argmax(profits))
    return float(grid[i]), float(profits[i]), grid, profits


def approve_all_profit(y_true, amount):
    """Baseline: aprobar todas las transacciones."""
    n = len(np.asarray(y_true))
    return total_profit(y_true, amount, np.ones(n, dtype=bool))


def reject_all_profit(y_true, amount):
    """Baseline: rechazar todas (siempre 0)."""
    return 0.0


def max_possible_profit(y_true, amount):
    """Techo teórico: aprobar solo las legítimas."""
    y_true = np.asarray(y_true)
    return total_profit(y_true, amount, y_true == 0)