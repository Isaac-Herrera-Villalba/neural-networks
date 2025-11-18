#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/nn/delta_rule.py
------------------------------------------------------------
Descripción:
Implementación EXACTA de la Regla Delta (Widrow–Hoff / ADALINE)
según el procedimiento mostrado en la presentación del curso de
Redes Neuronales.

Este modelo:
    - Usa NEURONA LINEAL (NO usa función escalón).
    - Producción de salida:
            y_hat = net = Σ (w_i * x_i)
    - Error lineal:
            e = y - y_hat
    - Actualización de pesos:
            w_i(new) = w_i(old) + η * e * x_i
    - Se minimiza el error cuadrático medio (MSE):
            MSE = (1/n) Σ e^2

Características del algoritmo:
------------------------------------------------------------
✓ Soporta n variables de entrada.
✓ Pesos iniciales opcionales.
✓ Procesamiento patrón por patrón (on-line).
✓ Registro detallado para generación de reportes LaTeX:
      - net
      - salida lineal y_hat
      - error e
      - delta_w
      - vectores de pesos antes/después
✓ Compatible con XOR, pero NO converge (linealmente inseparable).
✓ Compatible con AND, OR, ↔, etc.

Consumido por:
    - src/main.py
    - src/report/report_nn_builder.py
------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np


# ============================================================
# === ESTRUCTURA DE RESULTADO ================================
# ============================================================

class DeltaRuleResult:
    """
    Contenedor de los resultados del entrenamiento con Regla Delta.

    Atributos
    ---------
    epochs : int
        Número total de épocas ejecutadas.

    weights_initial : List[float]
        Pesos iniciales antes del entrenamiento.

    weights_final : List[float]
        Pesos finales tras el entrenamiento.

    mse_history : List[float]
        Historial del error cuadrático medio por época.

    epoch_history : List[Dict]
        Lista detallada de todas las operaciones realizadas:

        [
            {
                'epoch': n,
                'pattern_logs': [
                    {
                        'x': [...],
                        'y_target': valor,
                        'net': valor,
                        'y_hat': valor,
                        'error': valor,
                        'delta_w': [...],
                        'w_before': [...],
                        'w_after': [...]
                    }, ...
                ],
                'mse': valor
            },
            ...
        ]
    """

    def __init__(self,
                 epochs: int,
                 weights_initial: List[float],
                 weights_final: List[float],
                 mse_history: List[float],
                 epoch_history: List[Dict[str, Any]]):

        self.epochs = epochs
        self.weights_initial = weights_initial
        self.weights_final = weights_final
        self.mse_history = mse_history
        self.epoch_history = epoch_history


# ============================================================
# === FUNCIÓN PRINCIPAL DE ENTRENAMIENTO =====================
# ============================================================

def train_delta_rule(
    X: np.ndarray,
    Y: np.ndarray,
    learning_rate: float = 0.5,
    max_epochs: int = 50,
    error_threshold: float = 0.01,
    w_init: List[float] | None = None
) -> DeltaRuleResult:
    """
    Entrena una neurona lineal con la Regla Delta según:

        y_hat = net = Σ(w_i * x_i)
        e = y - y_hat
        w_i(new) = w_i(old) + η * e * x_i

    Parámetros
    ----------
    X : np.ndarray
        Matriz n_patrones × n_features

    Y : np.ndarray
        Vector objetivo (valores numéricos reales)

    learning_rate : float
        Tasa de aprendizaje η.

    max_epochs : int
        Número máximo de iteraciones sobre el dataset.

    error_threshold : float
        Criterio opcional de paro basado en MSE.

    w_init : List[float] | None
        Pesos iniciales opcionales.

    Retorna
    -------
    DeltaRuleResult
        Resultados completos del entrenamiento.
    """

    n_patterns, n_features = X.shape

    # --------------------------------------
    # Inicialización de pesos
    # --------------------------------------
    if w_init is None:
        w = np.zeros(n_features)
    else:
        w = np.array(w_init, dtype=float)

    weights_initial = w.copy().tolist()

    mse_history = []
    epoch_history: List[Dict[str, Any]] = []

    # --------------------------------------
    # Entrenamiento
    # --------------------------------------
    for epoch in range(1, max_epochs + 1):

        pattern_logs = []
        squared_errors = []

        for i in range(n_patterns):

            x = X[i]
            y_target = Y[i]

            # --- net ---
            net = float(np.dot(w, x))

            # salida lineal
            y_hat = net

            # error lineal
            error = y_target - y_hat

            # registrar error para MSE
            squared_errors.append(error ** 2)

            # Δw según regla delta
            delta_w = learning_rate * error * x

            w_before = w.copy().tolist()

            # actualización de pesos
            w = w + delta_w

            w_after = w.copy().tolist()

            # registrar patrón
            pattern_logs.append({
                "x": x.tolist(),
                "y_target": float(y_target),
                "net": net,
                "y_hat": float(y_hat),
                "error": float(error),
                "delta_w": delta_w.tolist(),
                "w_before": w_before,
                "w_after": w_after
            })

        # --------------------------------------
        # Error cuadrático medio por época
        # --------------------------------------
        mse = float(np.mean(squared_errors))
        mse_history.append(mse)

        epoch_history.append({
            "epoch": epoch,
            "pattern_logs": pattern_logs,
            "mse": mse
        })

        # criterio de paro
        if mse <= error_threshold:
            break

    # --------------------------------------
    # Fin del entrenamiento
    # --------------------------------------
    return DeltaRuleResult(
        epochs=epoch,
        weights_initial=weights_initial,
        weights_final=w.tolist(),
        mse_history=mse_history,
        epoch_history=epoch_history
    )

