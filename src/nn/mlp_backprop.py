#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/nn/mlp_backprop.py
------------------------------------------------------------
Implementación simple y didáctica de Backpropagation (MLP)
según la presentación de redes neuronales proporcionada.

Arquitectura fija:
        n entradas  →  H neuronas ocultas  →  1 salida

Funciones:
 - train_backpropagation(): entrenamiento estándar por épocas
 - Forward pass: usa sigmoide en capa oculta y salida
 - Backward pass: gradientes según la regla delta generalizada

Este módulo genera:
 - Historial de pesos por época
 - Historial de salidas de cada capa
 - MSE por época
 - Detección de convergencia opcional

Compatible con el generador LaTeX
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
    hidden_weights_history: list        # lista de matrices W_h por época
    output_weights_history: list        # lista de vectores W_o por época
    hidden_layer_history: list          # salidas de capa oculta
    output_history: list                # salidas finales por época
    mse_history: list                   # errores por época

    final_hidden_weights: np.ndarray    # pesos finales capa oculta
    final_output_weights: np.ndarray    # pesos finales capa salida
    converged: bool
    convergence_epoch: int


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

    Parámetros
    ----------
    X : np.ndarray (N x n)
    y : np.ndarray (N x 1)
    hidden_neurons : int
        Tamaño de la capa oculta.
    learning_rate : float
    max_epochs : int
    threshold : float
    initial_hidden_weights : matriz H x n (opcional)
    initial_output_weights : vector H (opcional)

    Retorna
    -------
    MLPBackpropResult
    """

    # Alinear forma
    y = y.reshape(-1, 1)
    N, n = X.shape

    # ----------------------------------------------
    # Inicialización de pesos
    # ----------------------------------------------
    if initial_hidden_weights is None:
        W_hidden = np.random.uniform(-0.5, 0.5, size=(hidden_neurons, n))
    else:
        W_hidden = np.array(initial_hidden_weights, dtype=float)

    if initial_output_weights is None:
        W_out = np.random.uniform(-0.5, 0.5, size=(hidden_neurons, 1))
    else:
        W_out = np.array(initial_output_weights, dtype=float).reshape(hidden_neurons, 1)

    # Historiales
    hidden_weights_history = []
    output_weights_history = []
    hidden_layer_history = []
    output_history = []
    mse_history = []

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
        net_h = X @ W_hidden.T                    # (N x H)
        out_h = sigmoid(net_h)                     # capa oculta

        net_o = out_h @ W_out                     # (N x 1)
        out_o = sigmoid(net_o)                    # salida final

        hidden_layer_history.append(out_h.copy())
        output_history.append(out_o.copy())

        # -------------------------
        #   ERROR Y MSE
        # -------------------------
        error = y - out_o
        mse = np.mean(error ** 2)
        mse_history.append(mse)

        if mse <= threshold:
            converged = True
            convergence_epoch = epoch
            break

        # -------------------------
        #   BACKPROPAGATION
        # -------------------------

        # Delta en capa de salida
        delta_o = error * sigmoid_derivative(out_o)      # (N x 1)

        # Delta en capa oculta
        delta_h = sigmoid_derivative(out_h) * (delta_o @ W_out.T)  # (N x H)

        # -------------------------
        #   ACTUALIZACIÓN DE PESOS
        # -------------------------
        # capa salida
        grad_out = out_h.T @ delta_o                     # (H x 1)
        W_out = W_out + learning_rate * grad_out

        # capa oculta
        grad_hidden = delta_h.T @ X                      # (H x n)
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

