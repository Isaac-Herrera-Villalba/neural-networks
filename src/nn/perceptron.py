#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/nn/perceptron.py
------------------------------------------------------------
Implementación del Perceptrón Simple (regla de aprendizaje
de Rosenblatt).

Este módulo realiza específicamente:
    - Inicialización de pesos (manual o aleatoria)
    - Entrenamiento ciclo por ciclo
    - Actualización con la regla:
          w = w + η * (y - ŷ) * x
    - Detección de convergencia
    - Registro de:
          * pesos por época
          * predicciones
          * errores
          * época de convergencia (si aplica)

Retorna un objeto PerceptronResult que será utilizado por
los módulos LaTeX para construir el reporte.
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
    weights_history: list        # lista de vectores de pesos por época
    predictions_history: list    # predicciones por época
    error_history: list          # errores por época (conteo de clasificaciones incorrectas)
    final_weights: np.ndarray    # pesos finales (vector)
    converged: bool              # True/False
    convergence_epoch: int       # época donde convergió (si aplica)


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
    Entrena un perceptrón simple.

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
        Umbral/bias fijo (se suma al producto punto).
    initial_weights : np.ndarray | None
        Pesos iniciales. Si None → aleatorios pequeños.

    Retorna
    -------
    PerceptronResult
    """

    N, n = X.shape

    # Aplanar y a vector 1D
    y = y.reshape(-1)

    # Inicialización de pesos
    if initial_weights is None:
        w = np.random.uniform(-0.5, 0.5, size=n)
    else:
        w = np.array(initial_weights, dtype=float).reshape(n)

    weights_history = []
    predictions_history = []
    error_history = []

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
            x_i = X[i]
            activation = np.dot(w, x_i) + threshold
            y_hat = 1 if activation >= 0 else -1
            y_pred_epoch.append(y_hat)

            if y_hat != y[i]:
                errors += 1
                # Regla del perceptrón
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

