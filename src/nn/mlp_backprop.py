#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/nn/mlp_backprop.py
------------------------------------------------------------
Descripción:
Implementación de un Perceptrón Multicapa (MLP) con aprendizaje
mediante Backpropagation EXACTAMENTE como se describe en la
presentación de Redes Neuronales entregada en el PDF.

Arquitectura soportada:
    - 1 capa oculta (neuronas configurables)
    - 1 neurona de salida
    - Activación sigmoide en TODAS las neuronas

El entrenamiento es ON-LINE (patrón por patrón), siguiendo:

    Forward-pass:
        net_j = Σ (w_ji * x_i)
        y_j   = sigmoid(net_j)

    Capa de salida:
        δ_k = (t_k - y_k) * y_k * (1 - y_k)

    Capa oculta:
        δ_j = y_j(1 - y_j) Σ_k( δ_k * w_jk )

    Actualización:
        w_ji(new) = w_ji(old) + η * δ_j * x_i
        w_kj(new) = w_kj(old) + η * δ_k * y_j

El módulo devuelve toda la traza completa:
    - nets, activaciones
    - deltas
    - errores
    - actualizaciones de pesos
    - pesos antes/después
    - historial por época y por patrón

Consumido por:
    - src/main.py
    - src/report/report_nn_builder.py
------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np


# ============================================================
# === Funciones auxiliares ===================================
# ============================================================

def sigmoid(x: float) -> float:
    """Función sigmoide estándar."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(y: float) -> float:
    """Derivada de la sigmoide usando la salida y."""
    return y * (1 - y)


# ============================================================
# === Contenedor de resultados ===============================
# ============================================================

class BackpropResult:
    """
    Contenedor completo del entrenamiento MLP.

    Atributos:
    ----------
    epochs : int
        Número real de épocas ejecutadas.

    W_input_hidden : np.ndarray
        Pesos finales entre entrada y capa oculta.

    W_hidden_output : np.ndarray
        Pesos finales entre capa oculta y salida.

    history : List[Dict]
        Registro completo para LaTeX:

        [
          {
            'epoch': n,
            'pattern_logs': [
                {
                  'x': [...],
                  'target': y,
                  'net_hidden': [...],
                  'y_hidden': [...],
                  'net_out': valor,
                  'y_out': valor,
                  'delta_out': valor,
                  'delta_hidden': [...],
                  'delta_w_ih': matriz,
                  'delta_w_ho': vector,
                  'w_ih_before': matriz,
                  'w_ho_before': vector,
                  'w_ih_after': matriz,
                  'w_ho_after': vector
                }
            ],
            'mse': valor
          }
        ]
    """

    def __init__(self,
                 epochs: int,
                 W_input_hidden: np.ndarray,
                 W_hidden_output: np.ndarray,
                 history: List[Dict[str, Any]]):

        self.epochs = epochs
        self.W_input_hidden = W_input_hidden
        self.W_hidden_output = W_hidden_output
        self.history = history


# ============================================================
# === Entrenamiento BACKPROPAGATION ==========================
# ============================================================

def train_backprop(
    X: np.ndarray,
    Y: np.ndarray,
    hidden_neurons: int = 2,
    learning_rate: float = 0.5,
    max_epochs: int = 10000,
    error_threshold: float = 0.01,
    w_init_ih: np.ndarray | None = None,
    w_init_ho: np.ndarray | None = None
) -> BackpropResult:
    """
    Entrena un MLP usando Backpropagation EXACTO del PDF.

    Parámetros
    ----------
    X : np.ndarray (n_patterns × n_features)
    Y : np.ndarray (n_patterns,)
    hidden_neurons : int
        Número de neuronas ocultas (configurable).
    learning_rate : float
    max_epochs : int
    error_threshold : float
    w_init_ih : pesos iniciales input→hidden (opcional)
    w_init_ho : pesos iniciales hidden→output (opcional)

    Retorna
    -------
    BackpropResult
    """

    n_patterns, n_features = X.shape

    # --------------------------------------------------------
    # Inicialización de pesos
    # --------------------------------------------------------
    if w_init_ih is None:
        W_ih = np.random.uniform(-0.5, 0.5, (hidden_neurons, n_features))
    else:
        W_ih = np.array(w_init_ih, dtype=float)

    if w_init_ho is None:
        W_ho = np.random.uniform(-0.5, 0.5, hidden_neurons)
    else:
        W_ho = np.array(w_init_ho, dtype=float)

    history = []

    # ========================================================
    # Entrenamiento por épocas
    # ========================================================
    for epoch in range(1, max_epochs + 1):

        pattern_logs = []
        squared_errors = []

        for i in range(n_patterns):

            x = X[i]
            target = Y[i]

            # ------------------------------------------------
            # 1. Forward pass
            # ------------------------------------------------

            # Capa oculta
            net_hidden = W_ih @ x               # vector
            y_hidden = np.array([sigmoid(n) for n in net_hidden])

            # Capa de salida (una sola neurona)
            net_out = np.dot(W_ho, y_hidden)
            y_out = sigmoid(net_out)

            # ------------------------------------------------
            # 2. Cálculo de errores / deltas
            # ------------------------------------------------
            error = target - y_out
            squared_errors.append(error**2)

            # δ_k (capa de salida)
            delta_out = error * sigmoid_derivative(y_out)

            # δ_j (capa oculta)
            delta_hidden = sigmoid_derivative(y_hidden) * (delta_out * W_ho)

            # ------------------------------------------------
            # 3. Cálculo de variaciones de pesos
            # ------------------------------------------------
            delta_W_ho = learning_rate * delta_out * y_hidden
            delta_W_ih = learning_rate * delta_hidden.reshape(-1, 1) @ x.reshape(1, -1)

            w_ih_before = W_ih.copy()
            w_ho_before = W_ho.copy()

            # Actualizar pesos
            W_ho = W_ho + delta_W_ho
            W_ih = W_ih + delta_W_ih

            w_ih_after = W_ih.copy()
            w_ho_after = W_ho.copy()

            # Registro del patrón
            pattern_logs.append({
                "x": x.tolist(),
                "target": float(target),
                "net_hidden": net_hidden.tolist(),
                "y_hidden": y_hidden.tolist(),
                "net_out": float(net_out),
                "y_out": float(y_out),
                "delta_out": float(delta_out),
                "delta_hidden": delta_hidden.tolist(),
                "delta_w_ih": delta_W_ih.tolist(),
                "delta_w_ho": delta_W_ho.tolist(),
                "w_ih_before": w_ih_before.tolist(),
                "w_ho_before": w_ho_before.tolist(),
                "w_ih_after": w_ih_after.tolist(),
                "w_ho_after": w_ho_after.tolist()
            })

        # --------------------------------------------------------
        # Error de la época
        # --------------------------------------------------------
        mse = float(np.mean(squared_errors))

        history.append({
            "epoch": epoch,
            "pattern_logs": pattern_logs,
            "mse": mse
        })

        # criterio de paro
        if mse <= error_threshold:
            break

    # --------------------------------------------------------
    # Retorno del resultado
    # --------------------------------------------------------
    return BackpropResult(
        epochs=epoch,
        W_input_hidden=W_ih,
        W_hidden_output=W_ho,
        history=history
    )

