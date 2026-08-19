"""Tests de la función de ganancia"""
import numpy as np
from src.profit import (
    BREAK_EVEN_THRESHOLD, transaction_value, total_profit,
    profit_from_threshold, find_best_threshold,
    approve_all_profit, reject_all_profit, max_possible_profit, captured_margin,
)


def test_umbral_equilibrio_es_020():
    assert abs(BREAK_EVEN_THRESHOLD - 0.20) < 1e-9


def test_legitima_aprobada_gana_25():
    assert total_profit([0], [100.0], [True]) == 25.0


def test_fraude_aprobado_pierde_todo():
    assert total_profit([1], [100.0], [True]) == -100.0


def test_rechazar_da_cero():
    assert reject_all_profit([0, 1], [100.0, 100.0]) == 0.0


def test_umbral_rechaza_fraude_probable():
    # legítima (p=0.01) aprobada, fraude (p=0.99) rechazado con umbral 0.20
    assert profit_from_threshold([0, 1], [100.0, 100.0], [0.01, 0.99], 0.20) == 25.0


def test_aprobar_todo():
    assert approve_all_profit([0, 1], [100.0, 100.0]) == -75.0


def test_ganancia_maxima():
    assert max_possible_profit([0, 1, 0], [100.0, 100.0, 40.0]) == 35.0


def test_mejor_umbral_separacion_perfecta():
    y = np.array([0, 0, 1, 1]); amt = np.array([100.0]*4)
    p = np.array([0.1, 0.1, 0.9, 0.9])
    _, mejor_ganancia, _, _ = find_best_threshold(y, amt, p)
    assert mejor_ganancia == 50.0


def test_captured_margin_oraculo_da_1_y_piso_da_0():
    y = [0, 1, 0]
    amt = [100.0, 100.0, 40.0]
    piso = approve_all_profit(y, amt)
    techo = max_possible_profit(y, amt)
    assert abs(captured_margin(techo, y, amt) - 1.0) < 1e-9
    assert abs(captured_margin(piso, y, amt) - 0.0) < 1e-9


def test_captured_margin_punto_medio():
    y = [0, 1, 0]
    amt = [100.0, 100.0, 40.0]
    piso = approve_all_profit(y, amt)
    techo = max_possible_profit(y, amt)
    medio = (piso + techo) / 2
    assert abs(captured_margin(medio, y, amt) - 0.5) < 1e-9