#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/nn/mlp_backprop.py
------------------------------------------------------------
Descripción general
------------------------------------------------------------
Implementación de un Perceptrón Multicapa (MLP) con una capa oculta,
entrenado mediante el algoritmo de **Backpropagation**, siguiendo las
fórmulas y el esquema conceptual presentados en el PDF de
"Neural Networks".

Modelo teórico (según la presentación)
------------------------------------------------------------
- El MLP se compone de:

    * Capa de entrada: vector x = (x_1, ..., x_n).
    * Capa oculta: neuronas con activación sigmoide.
    * Capa de salida: neuronas también con activación sigmoide.

- Se usa la función sigmoide clásica:

        σ(net) = 1 / (1 + e^{-net})

- La salida de la red para un patrón x se denota z = (z_1, ..., z_m),
  y el vector de objetivos asociados es t = (t_1, ..., t_m).

- Función de error (por patrón):

        E = Σ_i 1/2 (t_i - z_i)^2

  (es decir, suma de errores cuadráticos por neurona de salida).

- Derivadas parciales para la capa de salida (según el PDF):

        ∂E/∂v_i = (z_i − t_i) · z_i · (1 − z_i) · y_i

  donde:
    - v_i    : peso conectado a la neurona de salida i,
    - z_i    : salida de la neurona de salida i,
    - t_i    : objetivo correspondiente,
    - y_i    : salida de la neurona de la capa oculta que alimenta a v_i.

- Derivadas parciales para la primera capa oculta (según el PDF):

        ∂E/∂w_i = ( Σ_j ∂E/∂y_j ) · y_i · (1 − y_i) · x_i

  donde:
    - w_i    : peso conectado a la neurona de la capa oculta,
    - y_i    : salida de la neurona de la capa oculta,
    - x_i    : entrada correspondiente,
    - Σ_j    : suma de los términos propagados desde la capa de salida.

- Regla de actualización general:

        v_i ← v_i − η ∂E/∂v_i
        w_i ← w_i − η ∂E/∂w_i

  donde η es la tasa de aprendizaje.

Objetivo de este módulo
------------------------------------------------------------
Este archivo implementa:

  - Una clase de configuración `MLPConfig`, para especificar:
      * tasa de aprendizaje,
      * número de épocas,
      * tamaño de la capa oculta,
      * umbral de error para detener el entrenamiento, etc.

  - Una estructura `BackpropTrainingStep` que registra los valores
    relevantes de cada actualización de pesos (nets, salidas, deltas,
    pesos antes y después), útil para construir posteriormente tablas
    y explicaciones en LaTeX que sigan la lógica del PDF.

  - Una clase `MLPBackprop` que implementa:
      * inicialización de pesos para la capa oculta y de salida,
      * propagación hacia adelante (forward pass),
      * propagación hacia atrás del error (backward pass),
      * actualización de pesos patrón a patrón, conforme a:

            v_ij ← v_ij − η δ_k y_j
            w_ji ← w_ji − η δ_j x_i

        donde δ_k y δ_j son las “señales de error” de salida y de la
        capa oculta, respectivamente, definidas de acuerdo con las
        derivadas parciales del PDF.

No se utilizan matrices ni soluciones cerradas al estilo de la
regresión lineal; el algoritmo se implementa como un proceso
iterativo paso a paso, fiel al enfoque de Backpropagation del curso.
------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# === CONFIGURACIÓN DEL MLP ==================================
# ============================================================

@dataclass
class MLPConfig:
    """
    Parámetros de configuración para el entrenamiento de un MLP
    mediante Backpropagation.

    Atributos
    ---------
    n_hidden : int
        Número de neuronas en la capa oculta.

    learning_rate : float
        Tasa de aprendizaje η utilizada en las reglas de actualización
        de pesos:

            v_ij ← v_ij − η ∂E/∂v_ij
            w_ji ← w_ji − η ∂E/∂w_ji

    max_epochs : int
        Número máximo de épocas de entrenamiento.

    error_threshold : float
        Umbral para la suma (promedio) de errores por patrón. Si el
        error medio en una época desciende por debajo de este valor,
        el entrenamiento se detiene anticipadamente.

    random_state : Optional[int]
        Semilla opcional para la inicialización reproducible de pesos.
    """
    n_hidden: int = 2
    learning_rate: float = 0.5
    max_epochs: int = 5000
    error_threshold: float = 1e-2
    random_state: Optional[int] = None


# ============================================================
# === REGISTRO DE ENTRENAMIENTO ==============================
# ============================================================

@dataclass
class BackpropTrainingStep:
    """
    Registro detallado de una actualización de pesos en el MLP.

    Esta estructura permite documentar, para cada patrón y en cada
    época, el flujo esencial del algoritmo de backpropagation:

      - entrada x
      - nets y salidas de capa oculta y de salida
      - vectores de error y deltas
      - pesos antes y después de la actualización

    Atributos
    ---------
    epoch : int
        Época a la que pertenece este paso.

    pattern_index : int
        Índice del patrón dentro de la época.

    x : np.ndarray
        Vector de entrada (sin bias).

    target : np.ndarray
        Vector objetivo t para este patrón.

    net_hidden : np.ndarray
        Valores netos de la capa oculta antes de la sigmoide.

    out_hidden : np.ndarray
        Salidas de la capa oculta después de aplicar la función
        sigmoide. No incluye el bias; el bias se maneja aparte.

    net_output : np.ndarray
        Nets de la capa de salida.

    out_output : np.ndarray
        Salidas de la capa de salida (z).

    delta_output : np.ndarray
        Señales de error de la capa de salida, equivalentes a:

            δ_k = (z_k - t_k) z_k (1 - z_k)

    delta_hidden : np.ndarray
        Señales de error de la capa oculta, equivalentes (conceptualmente)
        a la fórmula del PDF en términos de ∂E/∂y_j y derivadas sigmoide.

    weights_input_hidden_before : np.ndarray
        Pesos de la capa entrada→oculta antes de la actualización.

    weights_input_hidden_after : np.ndarray
        Pesos de la capa entrada→oculta después de la actualización.

    weights_hidden_output_before : np.ndarray
        Pesos de la capa oculta→salida antes de la actualización.

    weights_hidden_output_after : np.ndarray
        Pesos de la capa oculta→salida después de la actualización.

    error_scalar : float
        Error escalar del patrón: E = Σ 1/2 (t_k - z_k)^2.
    """
    epoch: int
    pattern_index: int
    x: np.ndarray
    target: np.ndarray
    net_hidden: np.ndarray
    out_hidden: np.ndarray
    net_output: np.ndarray
    out_output: np.ndarray
    delta_output: np.ndarray
    delta_hidden: np.ndarray
    weights_input_hidden_before: np.ndarray
    weights_input_hidden_after: np.ndarray
    weights_hidden_output_before: np.ndarray
    weights_hidden_output_after: np.ndarray
    error_scalar: float


# ============================================================
# === FUNCIONES AUXILIARES ===================================
# ============================================================

def _sigmoid(net: np.ndarray) -> np.ndarray:
    """
    Función de activación sigmoide aplicada elemento a elemento:

        σ(net) = 1 / (1 + e^{-net})

    Parámetros
    ----------
    net : np.ndarray
        Arreglo de nets (capa oculta o de salida).

    Retorna
    -------
    np.ndarray
        Arreglo con σ(net) aplicado a cada componente.
    """
    return 1.0 / (1.0 + np.exp(-net))


def _sigmoid_derivative(out: np.ndarray) -> np.ndarray:
    """
    Derivada de la sigmoide en función de su propia salida:

        dσ/dnet = σ(net) (1 − σ(net))

    pero expresado como:

        σ'(out) = out (1 − out)

    Parámetros
    ----------
    out : np.ndarray
        Salidas de una capa (ya con sigmoide aplicada).

    Retorna
    -------
    np.ndarray
        Derivadas dσ/dnet evaluadas en cada componente.
    """
    return out * (1.0 - out)


# ============================================================
# === CLASE PRINCIPAL: MLP CON BACKPROP ======================
# ============================================================

class MLPBackprop:
    """
    Implementación de un MLP con una capa oculta, entrenado mediante
    Backpropagation siguiendo la formulación del PDF:

      - Unidades sigmoide en capas oculta y de salida.
      - Función de error E = Σ 1/2 (t_i - z_i)^2.
      - Derivadas parciales:

            ∂E/∂v_i = (z_i − t_i) z_i (1 − z_i) y_i
            ∂E/∂w_i = ( Σ_j ∂E/∂y_j ) y_i (1 − y_i) x_i

      - Actualización de cada peso:

            v_i ← v_i − η ∂E/∂v_i
            w_i ← w_i − η ∂E/∂w_i

    La implementación no utiliza álgebra matricial en el informe,
    pero internamente representa los pesos como arreglos NumPy
    (conceptualmente equivalentes a matrices) para facilitar los
    cálculos numéricos.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        config: Optional[MLPConfig] = None,
    ) -> None:
        """
        Inicializa la arquitectura del MLP:

          - n_inputs  : número de atributos de entrada.
          - n_hidden  : tomado de config.n_hidden.
          - n_outputs : número de neuronas de salida.

        Se añaden biases explícitamente:
          - Pesos entrada→oculta: shape (n_hidden, n_inputs + 1)
              (la última columna corresponde al bias para cada neurona oculta)
          - Pesos oculta→salida:  shape (n_outputs, n_hidden + 1)
              (la última columna corresponde al bias en la capa de salida)
        """
        if n_inputs <= 0:
            raise ValueError("n_inputs debe ser un entero positivo.")
        if n_outputs <= 0:
            raise ValueError("n_outputs debe ser un entero positivo.")

        self.config: MLPConfig = config or MLPConfig()
        if self.config.n_hidden <= 0:
            raise ValueError("n_hidden (en MLPConfig) debe ser un entero positivo.")

        self.n_inputs: int = n_inputs
        self.n_hidden: int = self.config.n_hidden
        self.n_outputs: int = n_outputs

        # Inicialización de pesos
        if self.config.random_state is not None:
            rng = np.random.default_rng(self.config.random_state)
            self.weights_input_hidden: np.ndarray = rng.uniform(
                -0.05, 0.05, size=(self.n_hidden, self.n_inputs + 1)
            )
            self.weights_hidden_output: np.ndarray = rng.uniform(
                -0.05, 0.05, size=(self.n_outputs, self.n_hidden + 1)
            )
        else:
            # Inicialización fija sencilla (p. ej., 0.2) como punto de partida
            self.weights_input_hidden = np.full(
                shape=(self.n_hidden, self.n_inputs + 1),
                fill_value=0.2,
                dtype=float,
            )
            self.weights_hidden_output = np.full(
                shape=(self.n_outputs, self.n_hidden + 1),
                fill_value=0.2,
                dtype=float,
            )

        # Historial completo de pasos de entrenamiento
        self.training_history: List[BackpropTrainingStep] = []

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Propagación hacia adelante para un solo patrón.

        Pasos:
          1) Extender la entrada con bias: x_ext = [x_1, ..., x_n, 1].
          2) Calcular nets de la capa oculta:
                net_hidden_j = Σ_i w_ji x_ext_i
          3) Aplicar sigmoide:
                out_hidden_j = σ(net_hidden_j)
          4) Extender out_hidden con bias: y_ext = [y_1, ..., y_h, 1].
          5) Calcular nets de la capa de salida:
                net_output_k = Σ_j v_kj y_ext_j
          6) Aplicar sigmoide:
                out_output_k = σ(net_output_k)

        Parámetros
        ----------
        x : np.ndarray
            Vector de entrada (sin bias), shape (n_inputs,).

        Retorna
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (net_hidden, out_hidden, net_output, out_output)
        """
        if x.shape[0] != self.n_inputs:
            raise ValueError(
                f"Se esperaban {self.n_inputs} entradas, pero se recibieron {x.shape[0]}."
            )

        # 1) Entrada extendida con bias
        x_ext = np.concatenate([x.astype(float), np.array([1.0])])  # shape (n_inputs + 1,)

        # 2) Nets de capa oculta
        #    net_hidden_j = Σ_i w_ji x_ext_i
        net_hidden = self.weights_input_hidden @ x_ext  # shape (n_hidden,)

        # 3) Salidas de capa oculta
        out_hidden = _sigmoid(net_hidden)               # shape (n_hidden,)

        # 4) Extender salidas ocultas con bias
        hidden_ext = np.concatenate([out_hidden, np.array([1.0])])  # (n_hidden + 1,)

        # 5) Nets de salida
        net_output = self.weights_hidden_output @ hidden_ext  # shape (n_outputs,)

        # 6) Salidas de salida
        out_output = _sigmoid(net_output)                     # shape (n_outputs,)

        return net_hidden, out_hidden, net_output, out_output

    # --------------------------------------------------------
    # BACKWARD PASS (CÁLCULO DE DELTAS)
    # --------------------------------------------------------

    def _backward(
        self,
        x: np.ndarray,
        target: np.ndarray,
        net_hidden: np.ndarray,
        out_hidden: np.ndarray,
        net_output: np.ndarray,
        out_output: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica las fórmulas de backpropagation para un patrón:

          - Capa de salida:
                δ_k = (z_k − t_k) z_k (1 − z_k)

          - Capa oculta:
                δ_j = ( Σ_k δ_k v_kj ) y_j (1 − y_j)

        donde v_kj son los pesos de la capa oculta→salida SIN contar
        el bias (es decir, solo las conexiones desde neuronas ocultas).

        Parámetros
        ----------
        x : np.ndarray
            Vector de entrada (sin bias).

        target : np.ndarray
            Vector objetivo t_k para el patrón.

        net_hidden : np.ndarray
            Nets de la capa oculta (no se usan directamente aquí,
            pero se incluyen por coherencia con la firma).

        out_hidden : np.ndarray
            Salidas de la capa oculta.

        net_output : np.ndarray
            Nets de la capa de salida.

        out_output : np.ndarray
            Salidas de la capa de salida (z_k).

        Retorna
        -------
        Tuple[np.ndarray, np.ndarray]
            (delta_output, delta_hidden)
        """
        # δ_k = (z_k − t_k) z_k (1 − z_k)
        diff = out_output - target
        delta_output = diff * _sigmoid_derivative(out_output)  # shape (n_outputs,)

        # Para la capa oculta:
        # δ_j = ( Σ_k δ_k v_kj ) y_j (1 − y_j)
        # NOTA: v_kj son los pesos que conectan la neurona oculta j con
        #       la neurona de salida k. Excluimos la columna de bias.
        v_no_bias = self.weights_hidden_output[:, :-1]  # shape (n_outputs, n_hidden)
        # Σ_k δ_k v_kj   →   vector shape (n_hidden,)
        propagated = delta_output @ v_no_bias           # (1 x n_outputs) * (n_outputs x n_hidden)
        delta_hidden = propagated * _sigmoid_derivative(out_hidden)

        return delta_output, delta_hidden

    # --------------------------------------------------------
    # ENTRENAMIENTO
    # --------------------------------------------------------

    def fit(self, X: np.ndarray, T: np.ndarray) -> List[BackpropTrainingStep]:
        """
        Entrena el MLP mediante Backpropagation, usando las fórmulas
        del PDF para las derivadas parciales y la actualización
        de pesos.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de shape (n_patrones, n_inputs).

        T : np.ndarray
            Matriz de objetivos de shape (n_patrones, n_outputs).

        Retorna
        -------
        List[BackpropTrainingStep]
            Historial completo de pasos de entrenamiento.
        """
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D.")
        if T.ndim != 2:
            raise ValueError("T debe ser una matriz 2D.")
        n_patrones, n_inputs = X.shape
        n_patrones_T, n_outputs = T.shape

        if n_inputs != self.n_inputs:
            raise ValueError(
                f"n_inputs en la red = {self.n_inputs}, pero X tiene {n_inputs} columnas."
            )
        if n_outputs != self.n_outputs:
            raise ValueError(
                f"n_outputs en la red = {self.n_outputs}, pero T tiene {n_outputs} columnas."
            )
        if n_patrones != n_patrones_T:
            raise ValueError(
                "Número de patrones de X y T no coincide: "
                f"{n_patrones} vs {n_patrones_T}."
            )

        eta = self.config.learning_rate
        self.training_history.clear()

        for epoch in range(self.config.max_epochs):
            epoch_errors: List[float] = []

            for idx in range(n_patrones):
                x_vec = X[idx, :].astype(float)
                t_vec = T[idx, :].astype(float)

                # -- Pesos antes de la actualización (copias para el registro) --
                w_in_hid_before = self.weights_input_hidden.copy()
                w_hid_out_before = self.weights_hidden_output.copy()

                # 1) FORWARD
                net_h, out_h, net_o, out_o = self._forward(x_vec)

                # Error escalar para el patrón: E = Σ 1/2 (t_k - z_k)^2
                error_vec = t_vec - out_o
                E_pattern = 0.5 * float(np.sum(error_vec ** 2))
                epoch_errors.append(E_pattern)

                # 2) BACKWARD: cálculo de deltas
                delta_out, delta_hid = self._backward(
                    x=x_vec,
                    target=t_vec,
                    net_hidden=net_h,
                    out_hidden=out_h,
                    net_output=net_o,
                    out_output=out_o,
                )

                # 3) ACTUALIZACIÓN DE PESOS
                #    Capa oculta→salida:
                #       v_ij ← v_ij − η δ_k y_j
                #    donde y_j son salidas ocultas o bias correspondiente.
                hidden_ext = np.concatenate([out_h, np.array([1.0])])  # (n_hidden + 1,)
                # Para cada neurona de salida k, y cada peso j:
                for k in range(self.n_outputs):
                    for j in range(self.n_hidden + 1):
                        grad_v_kj = delta_out[k] * hidden_ext[j]
                        self.weights_hidden_output[k, j] -= eta * grad_v_kj

                #    Capa entrada→oculta:
                #       w_ji ← w_ji − η δ_j x_i
                #    donde x_i son entradas o bias correspondiente.
                x_ext = np.concatenate([x_vec, np.array([1.0])])  # (n_inputs + 1,)
                for j in range(self.n_hidden):
                    for i in range(self.n_inputs + 1):
                        grad_w_ji = delta_hid[j] * x_ext[i]
                        self.weights_input_hidden[j, i] -= eta * grad_w_ji

                # -- Pesos después de la actualización --
                w_in_hid_after = self.weights_input_hidden.copy()
                w_hid_out_after = self.weights_hidden_output.copy()

                # 4) Registrar paso
                step = BackpropTrainingStep(
                    epoch=epoch,
                    pattern_index=idx,
                    x=x_vec.copy(),
                    target=t_vec.copy(),
                    net_hidden=net_h.copy(),
                    out_hidden=out_h.copy(),
                    net_output=net_o.copy(),
                    out_output=out_o.copy(),
                    delta_output=delta_out.copy(),
                    delta_hidden=delta_hid.copy(),
                    weights_input_hidden_before=w_in_hid_before,
                    weights_input_hidden_after=w_in_hid_after,
                    weights_hidden_output_before=w_hid_out_before,
                    weights_hidden_output_after=w_hid_out_after,
                    error_scalar=E_pattern,
                )
                self.training_history.append(step)

            # Criterio de parada por error medio
            mean_error = float(np.mean(epoch_errors)) if epoch_errors else 0.0
            if mean_error <= self.config.error_threshold:
                break

        return self.training_history

    # --------------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Propaga hacia adelante un conjunto de patrones y devuelve
        las salidas de la capa de salida.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada, shape (n_patrones, n_inputs).

        Retorna
        -------
        np.ndarray
            Matriz de salidas, shape (n_patrones, n_outputs).
        """
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D.")
        if X.shape[1] != self.n_inputs:
            raise ValueError(
                f"Se esperaban {self.n_inputs} columnas de entrada; "
                f"se recibieron {X.shape[1]}."
            )

        outputs = []
        for i in range(X.shape[0]):
            _, _, _, out_o = self._forward(X[i, :])
            outputs.append(out_o)
        return np.vstack(outputs)

