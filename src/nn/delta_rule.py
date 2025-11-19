#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/nn/delta_rule.py
------------------------------------------------------------
Implementación de la Regla Delta (ADALINE).

Este módulo utiliza la actualización basada en el error continuo:

        w = w + η * (y - ŷ) * x

donde ŷ = w · x + b (sin función escalón, aprendizaje lineal).

Características:
 - Compatible con entrenamiento de ADALINE según el PDF.
 - Permite pesos iniciales manuales.
 - Registra todo el historial:
        * pesos
        * errores cuadráticos
        * predicciones por época

 - No requiere que los datos sean linealmente separables.
   (En XOR demostrará explícitamente que NO converge).

Devuelve DeltaRuleResult para alimentar el generador LaTeX.
------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# ============================================================
# Resultado del entrenamiento Delta Rule
# ============================================================

@dataclass
class DeltaRuleResult:
    weights_history: list          # pesos por época
    predictions_history: list      # y_hat por época (continuo)
    mse_history: list              # lista del error cuadrático medio por época
    final_weights: np.ndarray      # pesos finales
    converged: bool                # si el error MSE alcanza un umbral
    convergence_epoch: int         # época de convergencia o -1


# ============================================================
# Entrenamiento Delta Rule (ADALINE)
# ============================================================

def train_delta_rule(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    max_epochs: int = 50,
    initial_weights: np.ndarray | None = None,
    threshold: float = 0.01
) -> DeltaRuleResult:
    """
    Entrena un modelo ADALINE usando la Regla Delta.

    Parámetros
    ----------
    X : np.ndarray (N x n)
        Datos de entrada.
    y : np.ndarray (N x 1)
        Vector objetivo con valores {-1, +1}.
    learning_rate : float
        Tasa de aprendizaje η.
    max_epochs : int
        Máximo número de épocas.
    initial_weights : np.ndarray | None
        Pesos iniciales.
    threshold : float
        MSE mínimo requerido para considerar convergencia.

    Retorna
    -------
    DeltaRuleResult
    """

    # Aplanar y
    y = y.reshape(-1)
    N, n = X.shape

    # inicialización de pesos
    if initial_weights is None:
        w = np.random.uniform(-0.5, 0.5, size=n)
    else:
        w = np.array(initial_weights, dtype=float).reshape(n)

    weights_history = []
    predictions_history = []
    mse_history = []

    converged = False
    convergence_epoch = -1

    # ========================================================
    # ITERAR ENTRENAMIENTO
    # ========================================================
    for epoch in range(max_epochs):

        weights_history.append(w.copy())

        # Predicciones continuas (sin función escalón)
        y_hat = X @ w
        predictions_history.append(y_hat.copy())

        # Error cuadrático medio
        mse = np.mean((y - y_hat) ** 2)
        mse_history.append(mse)

        # ¿Convergencia?
        if mse <= threshold:
            converged = True
            convergence_epoch = epoch
            break

        # Actualización de pesos (regla delta)
        # w = w + η * Σ (y - ŷ) x
        grad = (y - y_hat).reshape(-1, 1) * X
        w = w + learning_rate * grad.sum(axis=0)

    final_weights = w.copy()

    return DeltaRuleResult(
        weights_history=weights_history,
        predictions_history=predictions_history,
        mse_history=mse_history,
        final_weights=final_weights,
        converged=converged,
        convergence_epoch=convergence_epoch
    )

