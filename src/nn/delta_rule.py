#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/nn/delta_rule.py
------------------------------------------------------------
Implementación de la Regla Delta (ADALINE) con bias entrenable.

Modelo:
    o(x) = w0 * 1 + w1 x1 + ... + wn xn

- Objetivos en {0,1} (como en las diapositivas).
  Si el preprocesamiento trae y en {-1,1}, aquí se remapea a {0,1}.
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
    weights_history: list          # pesos (incluye bias) por época
    predictions_history: list      # y_hat por época (continuo)
    mse_history: list              # lista del error cuadrático medio por época
    final_weights: np.ndarray      # pesos finales (w0,...,wn)
    converged: bool                # si el error MSE alcanza un umbral
    convergence_epoch: int         # época de convergencia o -1


# ============================================================
# Utilidades internas
# ============================================================

def _augment_with_bias(X: np.ndarray) -> np.ndarray:
    """Añade columna x0 = 1 a la matriz de entrada X."""
    N = X.shape[0]
    bias = np.ones((N, 1), dtype=float)
    return np.hstack((bias, X.astype(float)))


def _to_zero_one_targets(y: np.ndarray) -> np.ndarray:
    """
    Convierte objetivos en {-1,1} a {0,1} si aplica.
    Si ya están en [0,1] se dejan igual.
    """
    y = y.reshape(-1).astype(float)
    unique = np.unique(y)
    if set(np.round(unique).tolist()) <= {-1.0, 1.0}:
        # mapeo estándar: -1 → 0, +1 → 1
        return (y + 1.0) / 2.0
    return y


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
        Vector objetivo (se normaliza internamente a {0,1}).
    learning_rate : float
        Tasa de aprendizaje η.
    max_epochs : int
        Máximo número de épocas.
    initial_weights : np.ndarray | None
        Pesos iniciales (w0,...,wn). Si se proporcionan sólo n pesos,
        se asume w0 = 0.
    threshold : float
        MSE mínimo requerido para considerar convergencia.

    Retorna
    -------
    DeltaRuleResult
    """

    # Objetivos en {0,1}
    y_target = _to_zero_one_targets(y)
    y_target = y_target.reshape(-1)

    # Augmentar entradas con bias
    Xb = _augment_with_bias(X)
    N, n_plus_1 = Xb.shape

    # inicialización de pesos
    if initial_weights is None:
        w = np.random.uniform(-0.5, 0.5, size=n_plus_1)
    else:
        iw = np.array(initial_weights, dtype=float).reshape(-1)
        if iw.size == n_plus_1:
            w = iw
        elif iw.size == n_plus_1 - 1:
            w = np.zeros(n_plus_1, dtype=float)
            w[1:] = iw
        else:
            raise ValueError(
                f"Dimensión de INITIAL_WEIGHTS incompatible: "
                f"se esperaban {n_plus_1} valores (o {n_plus_1 - 1} sin bias), "
                f"pero se recibieron {iw.size}."
            )

    weights_history: list[np.ndarray] = []
    predictions_history: list[np.ndarray] = []
    mse_history: list[float] = []

    converged = False
    convergence_epoch = -1

    # ========================================================
    # ITERAR ENTRENAMIENTO
    # ========================================================
    for epoch in range(max_epochs):

        weights_history.append(w.copy())

        # Predicciones continuas (sin función escalón)
        y_hat = Xb @ w
        predictions_history.append(y_hat.copy())

        # Error cuadrático medio
        mse = float(np.mean((y_target - y_hat) ** 2))
        mse_history.append(mse)

        # ¿Convergencia?
        if mse <= threshold:
            converged = True
            convergence_epoch = epoch
            break

        # Actualización de pesos (regla delta)
        # w = w + η * Σ (t - o) x
        grad = (y_target - y_hat).reshape(-1, 1) * Xb
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

