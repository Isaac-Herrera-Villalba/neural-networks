#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/nn/delta_rule.py
------------------------------------------------------------
Descripción general
------------------------------------------------------------
Implementación del modelo *unthresholded perceptron* y de la
**Regla Delta**, exactamente como se presenta en el PDF de
"Neural Networks".

Modelo teórico (según el PDF)
------------------------------------------------------------
A diferencia del perceptrón clásico con función signo, el modelo
delta utiliza una unidad lineal:

    o(x) = w_1 x_1 + w_2 x_2 + ... + w_n x_n

No existe función de activación tipo umbral.

La **regla delta** modifica los pesos de acuerdo con:

    Δw_i = η (t - o) x_i
    w_i  ← w_i + Δw_i

donde:
    η : tasa de aprendizaje
    t : salida objetivo
    o : salida actual del perceptrón (valor real)
    x_i : componente i del vector de entrada

Importante:
- NO se usa la función escalón ni la salida en {−1, +1}.
- Las salidas son valores reales.
- El método es apropiado para funciones linealmente aproximables
  como XOR, ↔ (doble implicación), etc., tal como se ve en el PDF.

Este módulo define:
    - DeltaRuleConfig: parámetros de entrenamiento.
    - DeltaTrainingStep: registro completo de cada iteración.
    - DeltaRule: clase principal que implementa entrenamiento y predicción.

Seguimiento EXACTO del PDF:
    Se implementa el caso descrito en las diapositivas:
    - Unidad lineal sin umbral.
    - Regla Delta clásica.
    - Actualizaciones patrón a patrón.
    - Iteraciones consecutivas tal como se muestra en el ejemplo XOR.
------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


# ============================================================
# === CONFIGURACIÓN DE ENTRENAMIENTO =========================
# ============================================================

@dataclass
class DeltaRuleConfig:
    """
    Configuración de entrenamiento para la Regla Delta.

    Atributos
    ---------
    learning_rate : float
        Tasa de aprendizaje η usada en:
            Δw_i = η (t - o) x_i

    max_epochs : int
        Máximo número de épocas. Cada época recorre todos los patrones
        en orden.

    stop_threshold : float
        Si |t - o| para todos los patrones es <= stop_threshold,
        el entrenamiento se detiene anticipadamente.

        Para reproducir fielmente ejemplos como XOR en el PDF,
        pueden usarse valores como 0.4, 0.5, etc.

    random_state : Optional[int]
        Semilla opcional para reproducibilidad en la inicialización
        de pesos.
    """
    learning_rate: float = 0.5
    max_epochs: int = 50
    stop_threshold: float = 0.01
    random_state: Optional[int] = None


# ============================================================
# === REGISTRO DETALLADO DE ENTRENAMIENTO ====================
# ============================================================

@dataclass
class DeltaTrainingStep:
    """
    Registro de una actualización de pesos aplicada con la Regla Delta.

    Atributos
    ---------
    epoch : int
        Época a la que pertenece este paso.

    pattern_index : int
        Índice del patrón dentro de la época.

    x : np.ndarray
        Vector de entrada [x_1, ..., x_n].

    net : float
        Valor de salida o = Σ w_i x_i.

    output : float
        Igual que net, por definición de unidad lineal.

    target : float
        Valor objetivo para el patrón.

    delta_w : np.ndarray
        Vector de actualizaciones Δw.

    weights_before : np.ndarray
        Pesos justo antes de aplicar Δw.

    weights_after : np.ndarray
        Pesos justo después de aplicar Δw.

    error_value : float
        Error simple del patrón: (t - o).

    Notes
    -----
    Este registro permite reconstruir iteración por iteración el
    ejemplo clásico del PDF para XOR, donde se muestra cómo cambian
    los pesos en cada paso y cómo las actualizaciones dependen del
    error actual y de la entrada x.
    """
    epoch: int
    pattern_index: int
    x: np.ndarray
    net: float
    output: float
    target: float
    delta_w: np.ndarray
    weights_before: np.ndarray
    weights_after: np.ndarray
    error_value: float


# ============================================================
# === CLASE PRINCIPAL: REGLA DELTA ===========================
# ============================================================

class DeltaRule:
    """
    Implementación EXACTA de la Regla Delta tal como se enseña en
    la presentación del profesor:

        o = Σ w_i x_i
        Δw_i = η (t - o) x_i

    Características clave:
      - NO usa función umbral.
      - NO usa activación sigmoide.
      - NO usa salida en {−1, +1}.
      - La salida es un valor real.
      - El entrenamiento es estrictamente patrón a patrón.

    Esta clase reproduce la filosofía del ejemplo XOR del PDF.
    """

    def __init__(self, n_features: int, config: Optional[DeltaRuleConfig] = None):
        """
        Inicialización del modelo delta.

        Parámetros
        ----------
        n_features : int
            Número de atributos de entrada x_i.

        config : Optional[DeltaRuleConfig]
            Configuración de aprendizaje. Si no se especifica,
            se utiliza un conjunto de hiperparámetros por defecto
            compatibles con los ejemplos del PDF (η = 0.5).
        """
        if n_features <= 0:
            raise ValueError("n_features debe ser un entero positivo.")

        self.config = config or DeltaRuleConfig()

        # Inicialización de pesos
        if self.config.random_state is not None:
            rng = np.random.default_rng(self.config.random_state)
            self.weights = rng.uniform(-0.05, 0.05, size=(n_features,))
        else:
            self.weights = np.full(shape=(n_features,), fill_value=0.2, dtype=float)

        self.training_history: List[DeltaTrainingStep] = []

    # --------------------------------------------------------
    # CÁLCULO DE SALIDA
    # --------------------------------------------------------

    def net_input(self, x: np.ndarray) -> float:
        """
        Calcula la salida lineal:

            o = Σ w_i x_i

        Parámetros
        ----------
        x : np.ndarray
            Vector de entrada del patrón actual.

        Retorna
        -------
        float
            Valor de salida o.
        """
        return float(np.dot(self.weights, x))

    # --------------------------------------------------------
    # ENTRENAMIENTO
    # --------------------------------------------------------

    def fit(self, X: np.ndarray, y: Sequence[float]) -> List[DeltaTrainingStep]:
        """
        Entrena el perceptrón lineal usando la regla delta.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de patrones de entrada de forma
                (num_patrones, num_atributos)

        y : Sequence[float]
            Lista de objetivos t_j para cada patrón.

        Retorna
        -------
        List[DeltaTrainingStep]
            Historial de cada actualización.
        """
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D.")
        num_patrones, num_atributos = X.shape
        if num_atributos != self.weights.size:
            raise ValueError("Dimensión de X incompatible con los pesos.")

        y_arr = np.asarray(list(y), dtype=float)
        if y_arr.shape[0] != num_patrones:
            raise ValueError("Len(y) != número de patrones en X.")

        eta = self.config.learning_rate
        self.training_history.clear()

        for epoch in range(self.config.max_epochs):
            errors_epoch = []

            for idx in range(num_patrones):
                x_vec = X[idx, :].astype(float)
                t = float(y_arr[idx])

                w_before = self.weights.copy()

                net_val = self.net_input(x_vec)
                o = net_val                        # Unidad lineal
                error = t - o

                # Δw_i = η (t - o) x_i
                delta_w = eta * error * x_vec

                # Actualización
                self.weights = self.weights + delta_w
                w_after = self.weights.copy()

                # Registrar
                step = DeltaTrainingStep(
                    epoch=epoch,
                    pattern_index=idx,
                    x=x_vec.copy(),
                    net=net_val,
                    output=o,
                    target=t,
                    delta_w=delta_w.copy(),
                    weights_before=w_before,
                    weights_after=w_after,
                    error_value=error,
                )
                self.training_history.append(step)

                errors_epoch.append(abs(error))

            # Criterio de parada
            if all(err <= self.config.stop_threshold for err in errors_epoch):
                break

        return self.training_history

    # --------------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la salida real del perceptrón lineal para cada patrón.

        Parámetros
        ----------
        X : np.ndarray
            Matriz (num_patrones, num_atributos).

        Retorna
        -------
        np.ndarray
            Vector de valores reales o(x).
        """
        return np.array([self.net_input(row) for row in X], dtype=float)

