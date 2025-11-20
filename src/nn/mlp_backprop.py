#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/nn/mlp_backprop.py
------------------------------------------------------------
Implementación de Backpropagation (MLP con 1 capa oculta)
según el esquema del PDF, pero:

- Soporta n entradas.
- Usa bias en:
    * Capa de entrada → capa oculta
    * Capa oculta → capa de salida
- Objetivos en {0,1}, con unidades sigmoide.
------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# ============================================================
#   Funciones auxiliares
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """x es la salida sigmoide ya activada → derivada rápida."""
    return x * (1.0 - x)


# ============================================================
#   Resultado del entrenamiento MLP Backprop
# ============================================================

@dataclass
class MLPBackpropResult:
    hidden_weights_history: list        # lista de matrices W_h por época (incluye bias)
    output_weights_history: list        # lista de vectores W_o por época (incluye bias)
    hidden_layer_history: list          # salidas de capa oculta (sin bias)
    output_history: list                # salidas finales por época
    mse_history: list                   # errores por época

    final_hidden_weights: np.ndarray    # pesos finales capa oculta
    final_output_weights: np.ndarray    # pesos finales capa salida
    converged: bool
    convergence_epoch: int


# ============================================================
#   Utilidades internas
# ============================================================

def _augment_with_bias(X: np.ndarray) -> np.ndarray:
    """Añade columna x0 = 1 a la matriz X."""
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
        return (y + 1.0) / 2.0
    return y


# ============================================================
#   ENTRENAMIENTO MLP BACKPROPAGATION
# ============================================================

def train_backpropagation(
    X: np.ndarray,
    y: np.ndarray,
    hidden_neurons: int = 2,
    learning_rate: float = 0.5,
    max_epochs: int = 500,
    threshold: float = 0.01,
    initial_hidden_weights: np.ndarray | None = None,
    initial_output_weights: np.ndarray | None = None,
) -> MLPBackpropResult:
    """
    Entrenamiento de una red neuronal MLP con backpropagation.

    Arquitectura:
        Entradas: n
        Capa oculta: H neuronas sigmoide (+ bias)
        Capa salida: 1 neurona sigmoide (+ bias)

    Parámetros
    ----------
    X : np.ndarray (N x n)
    y : np.ndarray (N x 1)
        Objetivos lógicos; si vienen en {-1,1}, se mapean a {0,1}.
    """

    # Objetivos en [0,1]
    y_target = _to_zero_one_targets(y).reshape(-1, 1)

    # Entrada con bias
    Xb = _augment_with_bias(X)
    N, n_plus_1 = Xb.shape

    # ----------------------------------------------
    # Inicialización de pesos
    # ----------------------------------------------
    # W_hidden: (H x (n+1))  → incluye bias w0_hj
    if initial_hidden_weights is None:
        W_hidden = np.random.uniform(-0.5, 0.5, size=(hidden_neurons, n_plus_1))
    else:
        W_hidden = np.array(initial_hidden_weights, dtype=float)
        if W_hidden.shape != (hidden_neurons, n_plus_1):
            raise ValueError(
                f"Dimensión de initial_hidden_weights debe ser "
                f"({hidden_neurons}, {n_plus_1}), pero es {W_hidden.shape}"
            )

    # W_out: ((H+1) x 1)  → incluye bias de la capa de salida
    if initial_output_weights is None:
        W_out = np.random.uniform(-0.5, 0.5, size=(hidden_neurons + 1, 1))
    else:
        W_out = np.array(initial_output_weights, dtype=float).reshape(-1, 1)
        if W_out.shape != (hidden_neurons + 1, 1):
            raise ValueError(
                f"Dimensión de initial_output_weights debe ser "
                f"({hidden_neurons + 1}, 1), pero es {W_out.shape}"
            )

    # Historiales
    hidden_weights_history: list[np.ndarray] = []
    output_weights_history: list[np.ndarray] = []
    hidden_layer_history: list[np.ndarray] = []
    output_history: list[np.ndarray] = []
    mse_history: list[float] = []

    converged = False
    convergence_epoch = -1

    # ========================================================
    #   INICIO DEL ENTRENAMIENTO
    # ========================================================
    for epoch in range(max_epochs):

        # Guardar pesos actuales
        hidden_weights_history.append(W_hidden.copy())
        output_weights_history.append(W_out.copy())

        # -------------------------
        #   FORWARD PASS
        # -------------------------
        net_h = Xb @ W_hidden.T              # (N x H)
        out_h = sigmoid(net_h)               # capa oculta (N x H)

        # añadir bias a la salida oculta
        out_hb = _augment_with_bias(out_h)   # (N x (H+1))

        net_o = out_hb @ W_out               # (N x 1)
        out_o = sigmoid(net_o)               # salida final (N x 1)

        hidden_layer_history.append(out_h.copy())
        output_history.append(out_o.copy())

        # -------------------------
        #   ERROR Y MSE
        # -------------------------
        error = y_target - out_o
        mse = float(np.mean(error ** 2))
        mse_history.append(mse)

        if mse <= threshold:
            converged = True
            convergence_epoch = epoch
            break

        # -------------------------
        #   BACKPROPAGATION
        # -------------------------

        # Delta en capa de salida
        delta_o = error * sigmoid_derivative(out_o)          # (N x 1)

        # Gradiente capa salida (incluye bias)
        grad_out = out_hb.T @ delta_o                        # (H+1 x 1)
        W_out = W_out + learning_rate * grad_out

        # Delta en capa oculta (ignorando el bias en la retropropagación)
        delta_h_raw = (delta_o @ W_out[1:].T) * sigmoid_derivative(out_h)  # (N x H)

        # Gradiente capa oculta (incluye bias hacia la entrada)
        grad_hidden = delta_h_raw.T @ Xb                     # (H x (n+1))
        W_hidden = W_hidden + learning_rate * grad_hidden

    # ========================================================
    #   RESULTADO FINAL
    # ========================================================
    return MLPBackpropResult(
        hidden_weights_history=hidden_weights_history,
        output_weights_history=output_weights_history,
        hidden_layer_history=hidden_layer_history,
        output_history=output_history,
        mse_history=mse_history,
        final_hidden_weights=W_hidden,
        final_output_weights=W_out,
        converged=converged,
        convergence_epoch=convergence_epoch
    )

