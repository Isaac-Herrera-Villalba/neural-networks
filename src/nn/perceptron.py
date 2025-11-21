#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/nn/perceptron.py
-------------------------------------------------------------------------------
Implementación del Perceptrón simple de Rosenblatt (aprendizaje supervisado)
-------------------------------------------------------------------------------

Descripción general
-------------------
Este módulo implementa el perceptrón clásico introducido por Rosenblatt,
siguiendo el modelo visto en la presentación:

        o(x) = sign( w0 + w1 x1 + ... + wn xn )

donde:
  • w0 es el **bias entrenable**, añadido internamente como una entrada fija 1.
  • Las salidas esperadas deben pertenecer al conjunto {-1, +1}.
  • El algoritmo actualiza los pesos únicamente cuando ocurre un error
    de clasificación (learning rule).

El perceptrón:
  - Soporta un número arbitrario de entradas (n características).
  - Utiliza aprendizaje iterativo por épocas.
  - Converge únicamente si el conjunto de datos es *linealmente separable*.
  - Se detiene cuando no hay errores en una época completa o cuando
    se alcanza max_epochs.

El módulo expone:
  1. La clase `PerceptronResult` (estructura con todo el historial generado).
  2. La función `train_perceptron()` (entrenamiento del modelo).
  3. La función auxiliar `_augment_with_bias()`.

Este módulo es invocado desde:
  - src/report/report_nn_builder.py (generación del bloque LaTeX)
  - src/main.py (ejecución del pipeline completo)
  - src/core/data_extractor.preprocess_numeric (entrada numérica X, y)

-------------------------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# =============================================================================
# Estructura del resultado del perceptrón
# =============================================================================

@dataclass
class PerceptronResult:
    """
    Objeto contenedor que almacena todos los datos generados durante
    el entrenamiento del perceptrón.

    Atributos
    ---------
    weights_history : list[np.ndarray]
        Lista con los pesos (incluido bias) en cada época.
        Cada elemento es un vector [w0, w1, ..., wn].

    predictions_history : list[np.ndarray]
        Predicciones realizadas en cada época, en forma de vector
        con valores {-1, +1}.

    error_history : list[int]
        Número total de errores cometidos en cada época.

    final_weights : np.ndarray
        Vector de pesos finales aprendidos por el perceptrón.

    converged : bool
        True si el perceptrón alcanzó 0 errores en alguna época.

    convergence_epoch : int
        Índice de la época donde ocurrió la convergencia (si ocurrió),
        o -1 si nunca convergió.
    """
    weights_history: list
    predictions_history: list
    error_history: list
    final_weights: np.ndarray
    converged: bool
    convergence_epoch: int


# =============================================================================
# Función interna: añadir bias a la matriz de entrada
# =============================================================================

def _augment_with_bias(X: np.ndarray) -> np.ndarray:
    """
    Añade una columna de bias a la matriz X.

    Esto implementa el truco estándar de Rosenblatt:
        x' = [1, x1, x2, ..., xn]

    Parámetros
    ----------
    X : np.ndarray (N x n)
        Matriz original de características.

    Retorna
    -------
    np.ndarray (N x (n+1))
        Matriz extendida con un bias fijo igual a 1 en la primera columna.
    """
    N = X.shape[0]
    bias = np.ones((N, 1), dtype=float)
    return np.hstack((bias, X.astype(float)))


# =============================================================================
# Entrenamiento del perceptrón simple
# =============================================================================

def train_perceptron(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    max_epochs: int = 20,
    threshold: float = 0.0,
    initial_weights: np.ndarray | None = None
) -> PerceptronResult:
    """
    Entrena un perceptrón simple de Rosenblatt utilizando la regla de aprendizaje:

        w  ←  w + η (y - y_hat) x'

    donde x' ya incluye el bias:
        x' = [1, x1, ..., xn]

    Parámetros
    ----------
    X : np.ndarray (N x n)
        Matriz de características numéricas.

    y : np.ndarray (N x 1)
        Vector objetivo con valores {-1, +1} según el modelo visto en clase.

    learning_rate : float
        Tasa de aprendizaje η.

    max_epochs : int
        Número máximo de épocas permitidas para el entrenamiento.

    threshold : float
        Término adicional sumado a la activación. Normalmente se mantiene en 0,
        ya que el perceptrón incorpora el bias en las entradas extendidas.

    initial_weights : np.ndarray | None
        Pesos iniciales opcionales.
        • Si None: se inicializan aleatoriamente.
        • Si tamaño n+1: se utiliza como [w0, ..., wn].
        • Si tamaño n: se asume w0=0 y se asignan pesos a w1...wn.

    Retorna
    -------
    PerceptronResult
        Estructura con historial completo y pesos finales.

    Notas
    -----
    • El perceptrón solo converge si la clase positiva y negativa son
      linealmente separables (diapositiva 6 del PDF).
    • Si no converge, se reporta converged=False sin lanzar excepciones.
    • La activación utilizada es: y_hat = sign(activation).
    """

    # Expandir entradas con bias fijo = 1
    Xb = _augment_with_bias(X)
    N, n_plus_1 = Xb.shape

    # Asegurar que y sea vector 1D
    y = y.reshape(-1)

    # Inicialización de pesos
    if initial_weights is None:
        w = np.random.uniform(-0.5, 0.5, size=n_plus_1)
    else:
        iw = np.array(initial_weights, dtype=float).reshape(-1)
        if iw.size == n_plus_1:
            w = iw
        elif iw.size == n_plus_1 - 1:
            # El usuario proporcionó solo w1..wn → añadir w0 = 0
            w = np.zeros(n_plus_1, dtype=float)
            w[1:] = iw
        else:
            raise ValueError(
                f"Dimensión de INITIAL_WEIGHTS incompatible: "
                f"se esperaban {n_plus_1} valores (o {n_plus_1 - 1} sin bias), "
                f"pero se recibieron {iw.size}."
            )

    # Historiales
    weights_history: list[np.ndarray] = []
    predictions_history: list[np.ndarray] = []
    error_history: list[int] = []

    converged = False
    convergence_epoch = -1

    # =========================================================================
    # Entrenamiento por épocas
    # =========================================================================
    for epoch in range(max_epochs):

        errors = 0
        y_pred_epoch = []

        # Guardar copia de los pesos actuales
        weights_history.append(w.copy())

        # Recorrido por patrones
        for i in range(N):
            x_i = Xb[i]
            activation = float(np.dot(w, x_i)) + float(threshold)
            y_hat = 1 if activation >= 0 else -1
            y_pred_epoch.append(y_hat)

            # Si hay error → aplicar regla de actualización
            if y_hat != y[i]:
                errors += 1
                w = w + learning_rate * (y[i] - y_hat) * x_i

        predictions_history.append(np.array(y_pred_epoch))
        error_history.append(errors)

        # Criterio de parada (convergencia)
        if errors == 0:
            converged = True
            convergence_epoch = epoch
            break

    # Pesos finales después del entrenamiento
    final_weights = w.copy()

    return PerceptronResult(
        weights_history=weights_history,
        predictions_history=predictions_history,
        error_history=error_history,
        final_weights=final_weights,
        converged=converged,
        convergence_epoch=convergence_epoch
    )

