#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/nn/perceptron.py
------------------------------------------------------------
Descripción:
Implementación exacta del algoritmo del Perceptrón
según la presentación de la práctica de Redes Neuronales.

El Perceptrón que se implementa aquí:

  - Opera sobre n variables de entrada (x1, x2, ..., xn).
  - Utiliza pesos iniciales w_i y un sesgo (bias) opcional.
  - Calcula el valor neto:
        net = Σ (w_i * x_i)
  - Aplica la función de activación escalón:
        y_pred = 1 si net >= 0
                 0 en otro caso
  - Calcula el error:
        e = y_target - y_pred
  - Ajusta pesos siguiendo la regla del perceptrón:
        w_i(new) = w_i(old) + η * e * x_i

Este módulo:
  ✓ No usa matrices.
  ✓ No usa derivadas.
  ✓ No usa conceptos que NO estén en el PDF.
  ✓ Sigue paso a paso la definición clásica del perceptrón.

El objetivo es entrenar la red hasta que:
  - TODOS los patrones se clasifiquen correctamente, O
  - se alcance MAX_EPOCHS.

Este módulo devuelve:
  - Pesos iniciales.
  - Lista completa de actualizaciones por época.
  - Pesos finales.
  - Indicador de convergencia.
  - Número total de épocas ejecutadas.
  - Registros completos para generar el reporte LaTeX.

Consumido por:
  - src/main.py
  - src/report/report_nn_builder.py
------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np


# ============================================================
# === ESTRUCTURA DE LA SALIDA ================================
# ============================================================

class PerceptronResult:
    """
    Contenedor de resultados del entrenamiento del perceptrón.

    Atributos
    ---------
    converged : bool
        True si el perceptrón clasificó correctamente todos los
        patrones antes de MAX_EPOCHS.

    epochs : int
        Número real de épocas ejecutadas.

    weights_initial : List[float]
        Pesos iniciales antes del entrenamiento.

    weights_final : List[float]
        Pesos finales después del entrenamiento.

    epoch_history : List[Dict]
        Lista ordenada de registros. Cada entrada contiene:

        {
            'epoch': número de época,
            'pattern_logs': [
                {
                    'x': [...],
                    'y_target': 0/1,
                    'net': valor,
                    'y_pred': 0/1,
                    'error': valor,
                    'delta_w': [...],
                    'w_before': [...],
                    'w_after': [...]
                },
                ...
            ]
        }
    """

    def __init__(self,
                 converged: bool,
                 epochs: int,
                 weights_initial: List[float],
                 weights_final: List[float],
                 epoch_history: List[Dict[str, Any]]):

        self.converged = converged
        self.epochs = epochs
        self.weights_initial = weights_initial
        self.weights_final = weights_final
        self.epoch_history = epoch_history


# ============================================================
# === FUNCIÓN PRINCIPAL DEL PERCEPTRÓN =======================
# ============================================================

def train_perceptron(
    X: np.ndarray,
    Y: np.ndarray,
    learning_rate: float = 0.5,
    max_epochs: int = 30,
    w_init: List[float] | None = None
) -> PerceptronResult:
    """
    Entrena un perceptrón según la regla mostrada en el PDF:

        net = Σ w_i * x_i
        y_pred = step(net)
        e = y - y_pred
        w_i(new) = w_i(old) + η * e * x_i

    Parámetros
    ----------
    X : np.ndarray, shape = (n_patrones, n_features)
        Matriz de patrones de entrada.
        Cada fila es un patrón: [x1, x2, ..., xn]

    Y : np.ndarray, shape = (n_patrones,)
        Vector de salidas objetivo (0/1)

    learning_rate : float
        Valor η usado en la actualización de pesos.

    max_epochs : int
        Número máximo de repeticiones sobre todo el dataset.

    w_init : list[float] | None
        Pesos iniciales opcionales.
        Si es None, se inicia con w_i = 0.0

    Retorna
    -------
    PerceptronResult
        Objeto con toda la información necesaria para generar
        el reporte LaTeX paso a paso.
    """

    n_patterns, n_features = X.shape

    # ----------------------------------------------
    # 1. Inicialización de pesos
    # ----------------------------------------------
    if w_init is None:
        w = np.zeros(n_features)
    else:
        w = np.array(w_init, dtype=float)

    weights_initial = w.copy().tolist()
    epoch_history: List[Dict[str, Any]] = []

    # ----------------------------------------------
    # 2. Entrenamiento por épocas
    # ----------------------------------------------
    for epoch in range(1, max_epochs + 1):

        pattern_logs = []
        errors_epoch = 0

        # recorrer todos los patrones
        for i in range(n_patterns):

            x = X[i]
            y_target = Y[i]

            # --- net ---
            net = float(np.dot(w, x))

            # --- activación escalón ---
            y_pred = 1 if net >= 0 else 0

            # --- error ---
            error = y_target - y_pred

            # registrar si hubo error
            if error != 0:
                errors_epoch += 1

            # --- cálculo de Δw ---
            delta_w = learning_rate * error * x

            w_before = w.copy().tolist()

            # actualizar pesos
            w = w + delta_w

            w_after = w.copy().tolist()

            # --- registrar patrón ---
            pattern_logs.append({
                "x": x.tolist(),
                "y_target": int(y_target),
                "net": net,
                "y_pred": int(y_pred),
                "error": int(error),
                "delta_w": delta_w.tolist(),
                "w_before": w_before,
                "w_after": w_after
            })

        # registrar la época completa
        epoch_history.append({
            "epoch": epoch,
            "pattern_logs": pattern_logs
        })

        # criterio de paro: no hubo errores
        if errors_epoch == 0:
            return PerceptronResult(
                converged=True,
                epochs=epoch,
                weights_initial=weights_initial,
                weights_final=w.tolist(),
                epoch_history=epoch_history
            )

    # ----------------------------------------------
    # 3. No convergió en MAX_EPOCHS
    # ----------------------------------------------
    return PerceptronResult(
        converged=False,
        epochs=max_epochs,
        weights_initial=weights_initial,
        weights_final=w.tolist(),
        epoch_history=epoch_history
    )

