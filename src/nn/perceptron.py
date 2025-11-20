#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/nn/perceptron.py
------------------------------------------------------------
Perceptrón simple de Rosenblatt con bias entrenable w0.

- Soporta n entradas (x1, ..., xn).
- Internamente se trabaja con un vector extendido:
      x' = [1, x1, ..., xn]
      w' = [w0, w1, ..., wn]

- Objetivos y ∈ {-1, +1}, como en el PDF.
------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# ============================================================
# Objeto de resultados del perceptrón
# ============================================================

@dataclass
class PerceptronResult:
    weights_history: list        # lista de vectores de pesos (incluye bias) por época
    predictions_history: list    # predicciones por época
    error_history: list          # errores por época (conteo de clasificaciones incorrectas)
    final_weights: np.ndarray    # pesos finales (w0,...,wn)
    converged: bool              # True/False
    convergence_epoch: int       # época donde convergió (si aplica)


# ============================================================
# Utilidad: añadir bias
# ============================================================

def _augment_with_bias(X: np.ndarray) -> np.ndarray:
    """
    Añade una columna x0 = 1 a la matriz de entrada X.

    X: (N x n) → Xb: (N x (n+1))
    """
    N = X.shape[0]
    bias = np.ones((N, 1), dtype=float)
    return np.hstack((bias, X.astype(float)))


# ============================================================
# Función principal de entrenamiento
# ============================================================

def train_perceptron(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    max_epochs: int = 20,
    threshold: float = 0.0,
    initial_weights: np.ndarray | None = None
) -> PerceptronResult:
    """
    Entrena un perceptrón simple con bias entrenable w0.

    Parámetros
    ----------
    X : np.ndarray (N x n)
        Matriz de entradas.
    y : np.ndarray (N x 1)
        Vector objetivo con valores {-1, +1}.
    learning_rate : float
        Tasa de aprendizaje η.
    max_epochs : int
        Número máximo de épocas.
    threshold : float
        Sesgo adicional fijo (normalmente 0.0). La recomendación es dejarlo en 0
        y usar sólo w0 como bias aprendible.
    initial_weights : np.ndarray | None
        Pesos iniciales. Si None → aleatorios pequeños.

        - Si tiene tamaño (n+1): se interpreta como [w0,...,wn].
        - Si tiene tamaño n: se asume que no se proporcionó bias, se usa w0=0.

    Retorna
    -------
    PerceptronResult
    """

    # Extender entradas con bias: x' = [1, x1, ..., xn]
    Xb = _augment_with_bias(X)
    N, n_plus_1 = Xb.shape

    # Aplanar y a vector 1D
    y = y.reshape(-1)

    # Inicialización de pesos (w0,...,wn)
    if initial_weights is None:
        w = np.random.uniform(-0.5, 0.5, size=n_plus_1)
    else:
        iw = np.array(initial_weights, dtype=float).reshape(-1)
        if iw.size == n_plus_1:
            w = iw
        elif iw.size == n_plus_1 - 1:
            # El usuario no proporcionó bias → inicializar w0 = 0
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
    error_history: list[int] = []

    converged = False
    convergence_epoch = -1

    # ========================================================
    # ITERAR ÉPOCA POR ÉPOCA
    # ========================================================
    for epoch in range(max_epochs):

        errors = 0
        y_pred_epoch = []

        # Guardar copia de pesos previos
        weights_history.append(w.copy())

        # Recorrer patrones
        for i in range(N):
            x_i = Xb[i]
            activation = float(np.dot(w, x_i)) + float(threshold)
            y_hat = 1 if activation >= 0 else -1
            y_pred_epoch.append(y_hat)

            if y_hat != y[i]:
                errors += 1
                # Regla del perceptrón (con x0=1 ya incluido en x_i)
                w = w + learning_rate * (y[i] - y_hat) * x_i

        predictions_history.append(np.array(y_pred_epoch))
        error_history.append(errors)

        # Condición de convergencia
        if errors == 0:
            converged = True
            convergence_epoch = epoch
            break

    # Guardar pesos finales
    final_weights = w.copy()

    return PerceptronResult(
        weights_history=weights_history,
        predictions_history=predictions_history,
        error_history=error_history,
        final_weights=final_weights,
        converged=converged,
        convergence_epoch=convergence_epoch
    )

