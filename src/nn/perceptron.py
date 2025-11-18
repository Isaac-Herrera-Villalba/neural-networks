#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/nn/perceptron.py
------------------------------------------------------------
Descripción general
------------------------------------------------------------
Implementación del perceptrón clásico de capa única tal como se
presenta en la práctica de *Redes Neuronales*.

Modelo teórico (según el PDF):

  - Entrada: vector x = (x_1, x_2, ..., x_n).
  - Se introduce un término de sesgo (bias) x_0 = 1.
  - Pesos: w = (w_0, w_1, ..., w_n).
  - Función de activación tipo umbral:

        o(x_1, ..., x_n) =
            1  si  w_0 + w_1 x_1 + ... + w_n x_n > 0
           -1  en otro caso

  - Regla de aprendizaje (regla del perceptrón):

        Δw_i = η (t - o) x_i
        w_i ← w_i + Δw_i

    donde:
      - η   : tasa de aprendizaje (learning rate)
      - t   : salida objetivo (target), en {−1, +1}
      - o   : salida actual del perceptrón para el patrón dado
      - x_i : i-ésima componente del vector de entrada (incluyendo x_0 = 1)

Este módulo proporciona:
  - Una clase de configuración (`PerceptronConfig`) con los parámetros
    de entrenamiento más relevantes.
  - Una estructura detallada para registrar cada paso de entrenamiento
    (`PerceptronTrainingStep`), útil para generar reportes en LaTeX
    con las iteraciones, tal como se muestra en los ejemplos del PDF.
  - Una clase `Perceptron` que encapsula:
        * inicialización de pesos,
        * cálculo de la salida o(x),
        * ciclo de entrenamiento patrón a patrón,
        * predicción sobre nuevos ejemplos.

Este módulo **no utiliza matrices ni librerías de ML externas**;
la actualización de pesos se implementa exactamente conforme a la
regla del perceptrón descrita en la presentación.
------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# === ESTRUCTURAS DE CONFIGURACIÓN Y TRAZAS ==================
# ============================================================

@dataclass
class PerceptronConfig:
    """
    Configuración del proceso de entrenamiento del perceptrón.

    Atributos
    ---------
    learning_rate : float
        Tasa de aprendizaje η utilizada en la regla de actualización:
            Δw_i = η (t - o) x_i.

        Valores típicos: 0.1, 0.01. En el PDF se usan valores de
        referencia como η = 0.1 para ejemplos pequeños.

    max_epochs : int
        Número máximo de pasadas completas sobre el conjunto de
        entrenamiento. Cada pasada sobre todos los patrones se
        considera una *época*.

    stop_when_no_errors : bool
        Si es True, el entrenamiento se detiene de forma anticipada
        cuando en una época completa no se comete ningún error
        (es decir, todos los patrones se clasifican correctamente).

    random_state : Optional[int]
        Semilla opcional para reproducir la inicialización aleatoria
        de los pesos. Si es None, se usa la inicialización por defecto
        de NumPy.
    """
    learning_rate: float = 0.1
    max_epochs: int = 50
    stop_when_no_errors: bool = True
    random_state: Optional[int] = None


@dataclass
class PerceptronTrainingStep:
    """
    Registro detallado de un único paso de actualización de pesos
    del perceptrón.

    Esta estructura está pensada para documentar iteraciones concretas
    en el reporte, de forma similar a como se presentan los ejemplos
    paso a paso en el PDF (AND, OR, etc.).

    Atributos
    ---------
    epoch : int
        Número de época (comenzando en 0). Una época corresponde a
        una pasada completa sobre todos los patrones disponibles.

    pattern_index : int
        Índice del patrón dentro del conjunto de entrenamiento
        (0, 1, ..., n-1) en la época actual.

    x : np.ndarray
        Vector de entrada extendido, incluyendo el bias:
            x = [x_0, x_1, ..., x_n]
        donde x_0 = 1.

    net : float
        Valor neto antes de la función de activación:
            net = w · x = w_0 x_0 + ... + w_n x_n

    output : int
        Salida del perceptrón tras aplicar la función de activación
        tipo umbral. Debe ser siempre 1 o -1, de acuerdo con:

            output = 1  si net > 0
                    -1  en otro caso

    target : int
        Valor objetivo t para el patrón actual. Debe ser 1 o -1.

    delta_w : np.ndarray
        Vector Δw resultante de aplicar la regla de aprendizaje:

            Δw_i = η (t - o) x_i

        Incluye también la componente Δw_0 del bias.

    weights_before : np.ndarray
        Copia de los pesos inmediatamente antes de aplicar la
        actualización en este patrón.

    weights_after : np.ndarray
        Copia de los pesos inmediatamente después de aplicar la
        actualización.

    misclassified : bool
        Indica si el patrón estaba mal clasificado antes de la
        actualización. Es True si t != o.

    Notes
    -----
    - El historial de estos pasos puede usarse posteriormente para
      generar tablas en LaTeX que muestren, por ejemplo:

          Época, patrón, x, net, o, t, Δw, w(antes), w(después)

      imitando exactamente el estilo de la práctica.
    """
    epoch: int
    pattern_index: int
    x: np.ndarray
    net: float
    output: int
    target: int
    delta_w: np.ndarray
    weights_before: np.ndarray
    weights_after: np.ndarray
    misclassified: bool


# ============================================================
# === FUNCIONES AUXILIARES ===================================
# ============================================================

def encode_binary_targets(y: Iterable[int | float]) -> np.ndarray:
    """
    Convierte etiquetas binarias en el rango {0, 1} o {−1, +1} al
    formato requerido {−1, +1}.

    Reglas:
      - Si el conjunto de valores es {0, 1}, se mapea:
            0 -> -1
            1 -> +1
      - Si el conjunto de valores es {−1, +1}, se dejan igual.
      - Cualquier otro valor provoca un ValueError.

    Parámetros
    ----------
    y : Iterable[int | float]
        Secuencia de etiquetas originales.

    Retorna
    -------
    np.ndarray
        Vector 1D de enteros en {−1, +1}.

    Excepciones
    -----------
    ValueError
        Si se encuentran etiquetas fuera del conjunto permitido.
    """
    arr = np.asarray(list(y), dtype=float)
    unique_vals = np.unique(arr)

    # Caso {0, 1}
    if np.array_equal(unique_vals, np.array([0.0, 1.0])) or np.array_equal(unique_vals, np.array([0.0])) \
       or np.array_equal(unique_vals, np.array([1.0])):
        return np.where(arr > 0.5, 1, -1).astype(int)

    # Caso {-1, 1}
    if np.all(np.isin(unique_vals, [-1.0, 1.0])):
        return arr.astype(int)

    raise ValueError(
        f"Las etiquetas deben estar en {{0,1}} o {{-1,+1}}. "
        f"Se encontraron valores: {unique_vals}"
    )


def _add_bias_column(X: np.ndarray) -> np.ndarray:
    """
    Añade la columna de bias (x_0 = 1) a una matriz de patrones.

    Dado un conjunto de patrones de entrada:

        X =
          [x_11  x_12  ...  x_1n]
          [x_21  x_22  ...  x_2n]
          [ ...              ...]
          [x_m1  x_m2  ...  x_mn]

    esta función devuelve:

        X_ext =
          [1  x_11  x_12  ...  x_1n]
          [1  x_21  x_22  ...  x_2n]
          [1  x_m1  x_m2  ...  x_mn]

    Parámetros
    ----------
    X : np.ndarray
        Matriz de entrada de forma (num_patrones, num_atributos).

    Retorna
    -------
    np.ndarray
        Matriz extendida con una primera columna de unos (bias).
    """
    if X.ndim != 2:
        raise ValueError(
            f"Se esperaba una matriz 2D para X, pero se recibió un arreglo "
            f"con ndim={X.ndim}."
        )
    num_patrones: int = X.shape[0]
    bias = np.ones((num_patrones, 1), dtype=float)
    return np.hstack([bias, X.astype(float)])


# ============================================================
# === CLASE PRINCIPAL: PERCEPTRÓN ============================
# ============================================================

class Perceptron:
    """
    Implementación del perceptrón de capa única con función de
    activación tipo umbral y regla de entrenamiento del PDF.

    La clase se encarga de:

      - Gestionar el vector de pesos w = [w_0, ..., w_n].
      - Realizar el cálculo de la salida o(x) para un patrón dado.
      - Ejecutar el ciclo de entrenamiento por épocas, recorriendo
        todos los patrones del conjunto de entrenamiento.
      - Registrar las actualizaciones de pesos en un historial
        detallado (`training_history`), útil para generar el reporte.

    A diferencia de un enfoque con álgebra matricial, aquí se sigue
    fielmente el esquema patrón a patrón descrito en las diapositivas,
    aplicando:

        Δw_i = η (t - o) x_i
        w_i ← w_i + Δw_i

    después de procesar cada patrón.
    """

    def __init__(self, n_features: int, config: Optional[PerceptronConfig] = None) -> None:
        """
        Inicializa un perceptrón con un número fijo de atributos
        de entrada (sin contar el bias).

        Parámetros
        ----------
        n_features : int
            Número de atributos de entrada (x_1, ..., x_n). El bias
            x_0 = 1 se maneja de forma interna y no se cuenta aquí.

        config : Optional[PerceptronConfig]
            Objeto de configuración con la tasa de aprendizaje, número
            máximo de épocas, etc. Si es None, se usa una configuración
            por defecto (learning_rate=0.1, max_epochs=50, ...).
        """
        if n_features <= 0:
            raise ValueError("n_features debe ser un entero positivo.")

        self.config: PerceptronConfig = config or PerceptronConfig()

        # Inicialización de pesos: vector [w_0, w_1, ..., w_n]
        # Se usan pequeños valores aleatorios o ceros (según preferencia).
        if self.config.random_state is not None:
            rng = np.random.default_rng(self.config.random_state)
            self.weights: np.ndarray = rng.uniform(-0.05, 0.05, size=(n_features + 1,))
        else:
            # Opción simple: inicializar todos los pesos con un valor pequeño fijo.
            self.weights = np.full(shape=(n_features + 1,), fill_value=0.2, dtype=float)

        # Historial completo de pasos de entrenamiento
        self.training_history: List[PerceptronTrainingStep] = []

    # --------------------------------------------------------
    # PROPIEDADES DE SOLO LECTURA
    # --------------------------------------------------------

    @property
    def n_features(self) -> int:
        """
        Número de atributos de entrada (sin contar el bias).

        Este valor es coherente con el tamaño del vector de pesos:
            len(weights) = n_features + 1.
        """
        return self.weights.size - 1

    # --------------------------------------------------------
    # CÁLCULO DE SALIDA
    # --------------------------------------------------------

    def net_input(self, x_with_bias: np.ndarray) -> float:
        """
        Calcula el valor neto `net = w · x` para un vector de entrada
        que ya incluye la componente de bias.

        Parámetros
        ----------
        x_with_bias : np.ndarray
            Vector 1D de longitud n_features + 1, con:
                x_with_bias[0] = 1  (bias)
                x_with_bias[1:] = [x_1, ..., x_n]

        Retorna
        -------
        float
            Valor neto antes de la función de activación.
        """
        if x_with_bias.shape != self.weights.shape:
            raise ValueError(
                f"Dimensiones incompatibles: x tiene shape {x_with_bias.shape}, "
                f"pero se esperaban {self.weights.shape}."
            )
        return float(np.dot(self.weights, x_with_bias))

    def activation(self, net: float) -> int:
        """
        Función de activación del perceptrón:

            o = 1   si net > 0
                -1  en otro caso

        Esta función implementa exactamente la definición de la
        diapositiva del PDF: una unidad umbral con salidas en {−1, +1}.

        Parámetros
        ----------
        net : float
            Valor neto w · x.

        Retorna
        -------
        int
            Salida o del perceptrón, en {−1, +1}.
        """
        return 1 if net > 0.0 else -1

    # --------------------------------------------------------
    # ENTRENAMIENTO
    # --------------------------------------------------------

    def fit(self, X: np.ndarray, y: Sequence[int | float]) -> List[PerceptronTrainingStep]:
        """
        Entrena el perceptrón sobre un conjunto de patrones de entrada
        y sus correspondientes etiquetas objetivo, siguiendo la regla
        del perceptrón del PDF.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (num_patrones, num_atributos),
            sin incluir el bias. Cada fila es un patrón:

                X[i, :] = [x_1, ..., x_n]_i

        y : Sequence[int | float]
            Secuencia de etiquetas asociadas a cada patrón. Puede venir
            en {0, 1} o en {−1, +1}; internamente se mapea siempre a
            {−1, +1} mediante `encode_binary_targets`.

        Retorna
        -------
        List[PerceptronTrainingStep]
            Lista con el historial completo de pasos de actualización
            de pesos realizados durante el entrenamiento. Este mismo
            objeto se almacena en `self.training_history`.

        Notas
        -----
        - El entrenamiento se realiza por épocas: en cada época se
          recorren todos los patrones, en el orden en que aparecen.
        - Por cada patrón, se calcula:

              net  = w · x
              o    = activación(net)
              Δw_i = η (t - o) x_i
              w_i  ← w_i + Δw_i

          donde x_0 = 1 se añade internamente.
        - Si `stop_when_no_errors` es True, el bucle se detiene
          cuando en una época no se comete ningún error.
        """
        # Validar dimensiones de X
        if X.ndim != 2:
            raise ValueError(
                f"X debe ser una matriz 2D de forma (num_patrones, num_atributos), "
                f"pero se recibió ndim={X.ndim}."
            )
        num_patrones, num_atributos = X.shape
        if num_atributos != self.n_features:
            raise ValueError(
                f"El perceptrón fue inicializado con n_features={self.n_features}, "
                f"pero X tiene {num_atributos} columnas."
            )

        # Normalizar etiquetas a {−1, +1}
        targets = encode_binary_targets(y)
        if targets.shape[0] != num_patrones:
            raise ValueError(
                "La cantidad de etiquetas no coincide con el número de patrones: "
                f"{targets.shape[0]} etiquetas para {num_patrones} patrones."
            )

        X_ext = _add_bias_column(X)  # añade x_0 = 1
        eta: float = self.config.learning_rate
        self.training_history.clear()

        for epoch in range(self.config.max_epochs):
            errores_en_epoca = 0

            for idx in range(num_patrones):
                x_vec: np.ndarray = X_ext[idx, :]
                t: int = int(targets[idx])

                w_before = self.weights.copy()
                net_val: float = self.net_input(x_vec)
                o: int = self.activation(net_val)

                # Diferencia (t - o) determina si hay error
                diff: int = t - o

                # Regla de actualización del perceptrón:
                # Δw_i = η (t - o) x_i
                delta_w: np.ndarray = eta * diff * x_vec

                # Actualizar pesos
                self.weights = self.weights + delta_w

                misclassified: bool = (diff != 0)
                if misclassified:
                    errores_en_epoca += 1

                step = PerceptronTrainingStep(
                    epoch=epoch,
                    pattern_index=idx,
                    x=x_vec.copy(),
                    net=net_val,
                    output=o,
                    target=t,
                    delta_w=delta_w.copy(),
                    weights_before=w_before,
                    weights_after=self.weights.copy(),
                    misclassified=misclassified,
                )
                self.training_history.append(step)

            # Criterio de parada por convergencia
            if self.config.stop_when_no_errors and errores_en_epoca == 0:
                break

        return self.training_history

    # --------------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------------

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula las salidas netas (w · x) para un conjunto de patrones,
        sin aplicar la función de activación.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (num_patrones, num_atributos),
            sin columna de bias.

        Retorna
        -------
        np.ndarray
            Vector 1D con los valores netos para cada patrón.
        """
        if X.ndim != 2:
            raise ValueError(
                f"X debe ser una matriz 2D; ndim={X.ndim} recibido."
            )
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X debe tener {self.n_features} columnas (atributos). "
                f"Se recibieron {X.shape[1]}."
            )

        X_ext = _add_bias_column(X)
        return np.dot(X_ext, self.weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula las salidas clasificadas del perceptrón para un
        conjunto de patrones de entrada, aplicando la función
        de activación umbral.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (num_patrones, num_atributos),
            sin incluir el bias.

        Retorna
        -------
        np.ndarray
            Vector 1D de enteros en {−1, +1}, correspondientes a la
            salida o(x) de cada patrón.

        Notas
        -----
        Esta operación NO modifica los pesos; únicamente se utiliza
        el vector de pesos ya entrenado.
        """
        net_vals = self.predict_raw(X)
        outputs = np.where(net_vals > 0.0, 1, -1).astype(int)
        return outputs

