#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/report/latex_backprop.py
------------------------------------------------------------
Descripción:
Generador del bloque LaTeX correspondiente al entrenamiento
de un Perceptrón Multicapa (MLP) entrenado mediante el método
de Backpropagation, siguiendo EXCLUSIVAMENTE el procedimiento
descrito en la presentación de Redes Neuronales.

Características:
------------------------------------------------------------
✓ Dos capas:
      - Capa oculta con n_h neuronas.
      - Capa de salida con 1 neurona.

✓ Fórmulas EXACTAS del PDF:
      net_h      = Σ w_ih * x_i
      y_h        = f(net_h)
      net_o      = Σ w_ho * y_h
      y_hat      = f(net_o)
      error e    = y - y_hat
      δ_o        = f'(net_o) * e
      δ_h        = f'(net_h) * Σ (δ_o * w_ho)

✓ Actualización de pesos:
      Δw_ho = η * δ_o * y_h
      Δw_ih = η * δ_h * x_i

✓ MSE por época:
      MSE = (1/N) Σ (y - y_hat)^2

✓ Tablas por época y por patrón.

Consumido por:
  - src/report/report_nn_builder.py

Requiere:
  - src/nn/mlp_backprop.py (MLPBackpropResult)
  - src/report/report_latex.py (_fmt_number, dataset_preview_table)
------------------------------------------------------------
"""

from __future__ import annotations
from typing import List
import pandas as pd

from src.nn.mlp_backprop import MLPBackpropResult
from src.report.report_latex import _fmt_number, dataset_preview_table


# ============================================================
# === Utilidades internas ====================================
# ============================================================

def _fmt_vector(v: List[float]) -> str:
    """Formato (v1, v2, v3) amigable para LaTeX."""
    return "(" + ", ".join(_fmt_number(x) for x in v) + ")"


def _fmt_matrix(mat: List[List[float]]) -> str:
    """Convierte una matriz a LaTeX tipo (fila1 ; fila2 ; ...)."""
    rows = ["(" + ", ".join(_fmt_number(v) for v in row) + ")" for row in mat]
    return "( " + " ; ".join(rows) + " )"


# ============================================================
# === Bloque Backpropagation =================================
# ============================================================

def build_backprop_block(
    df_num: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    learning_rate: float,
    hidden_neurons: int,
    max_epochs: int,
    error_threshold: float,
    result: MLPBackpropResult,
    activation_name: str,
    section_title: str | None = None,
) -> str:
    """
    Construye el bloque LaTeX completo correspondiente al entrenamiento
    de un MLP utilizando Backpropagation, siguiendo exactamente el método
    del PDF.

    Parámetros
    ----------
    df_num : pd.DataFrame
        Datos numéricos.
    x_cols : List[str]
        Entradas X.
    y_col : str
        Salida objetivo Y.
    learning_rate : float
        Tasa de aprendizaje η.
    hidden_neurons : int
        Número de neuronas en la capa oculta.
    max_epochs : int
        Número máximo de épocas permitidas.
    error_threshold : float
        Umbral mínimo para detener entrenamiento.
    result : MLPBackpropResult
        Resultado devuelto por train_mlp_backprop().
    activation_name : str
        Nombre de la función de activación (e.g. SIGMOID).
    section_title : str | None
        Título de la sección en el PDF.

    Retorna
    -------
    str
        Bloque LaTeX formateado.
    """
    lines: List[str] = []

    if section_title is None:
        section_title = (
            f"Backpropagation para la función {y_col} "
            f"con entradas ({', '.join(x_cols)})"
        )

    # ========================================================
    # 1. Encabezado
    # ========================================================
    lines.append(f"\\section*{{{section_title}}}")
    lines.append(
        r"En esta sección se documenta paso a paso el entrenamiento "
        r"de un perceptrón multicapa (MLP) de una sola capa oculta, "
        r"empleando el algoritmo de \textbf{Backpropagation}."
    )

    lines.append(r"\subsection*{Resumen del dataset}")
    lines.append(dataset_preview_table(df_num))

    # ========================================================
    # 2. Modelo MLP según el PDF
    # ========================================================
    lines.append(r"\subsection*{Modelo MLP y ecuaciones del PDF}")

    # Capa oculta
    lines.append(r"\textbf{Capa oculta:}")
    lines.append(r"\[ \text{net}_h = \sum_{i=1}^{n} w_{ih} x_i \]")
    lines.append(r"\[ y_h = f(\text{net}_h) \]")

    # Capa salida
    lines.append(r"\textbf{Capa de salida:}")
    lines.append(r"\[ \text{net}_o = \sum_{h=1}^{H} w_{ho} y_h \]")
    lines.append(r"\[ \hat{y} = f(\text{net}_o) \]")

    # Error y deltas
    lines.append(r"\textbf{Error y términos de corrección (PDF):}")
    lines.append(r"\[ e = y - \hat{y} \]")
    lines.append(r"\[ \delta_o = f'(\text{net}_o) \; e \]")
    lines.append(
        r"\[ \delta_h = f'(\text{net}_h)\sum_{o} \delta_o w_{ho} \]"
        r" \hspace{2em} \text{(solo hay una neurona de salida)}"
    )

    # Actualizaciones
    lines.append(r"\textbf{Actualización de pesos:}")
    lines.append(r"\[ \Delta w_{ho} = \eta \, \delta_o \, y_h \]")
    lines.append(r"\[ \Delta w_{ih} = \eta \, \delta_h \, x_i \]")

    # ========================================================
    # 3. Parámetros del entrenamiento
    # ========================================================
    lines.append(r"\subsection*{Parámetros de entrenamiento}")

    lines.append(r"\begin{itemize}")
    lines.append(rf"\item Tasa de aprendizaje: $\eta = {_fmt_number(learning_rate)}$.")
    lines.append(rf"\item Neuronas ocultas: $H = {hidden_neurons}$.")
    lines.append(rf"\item Épocas máximas: $N_{{\text{{max}}}} = {max_epochs}$.")
    lines.append(rf"\item Umbral de error: $\epsilon = {_fmt_number(error_threshold)}$.")
    lines.append(rf"\item Activación: \texttt{{{activation_name}}}.")
    lines.append(rf"\item Pesos iniciales capa oculta: ${_fmt_matrix(result.weights_hidden_initial)}$.")
    lines.append(rf"\item Pesos iniciales capa salida: ${_fmt_vector(result.weights_output_initial)}$.")
    lines.append(r"\end{itemize}")

    # ========================================================
    # 4. Desarrollo del entrenamiento
    # ========================================================
    lines.append(r"\subsection*{Desarrollo del entrenamiento (época por época)}")

    final_mse = result.mse_history[-1]
    if final_mse <= error_threshold:
        lines.append(
            rf"El MLP alcanza el criterio de paro en la época $E = {result.epochs}$ "
            rf"con un MSE final de ${_fmt_number(final_mse)}$."
        )
    else:
        lines.append(
            r"El entrenamiento no alcanza el umbral de error especificado; "
            r"se muestran los cálculos completos."
        )

    # Recorrer cada época
    for epoch_entry in result.epoch_history:
        epoch = epoch_entry["epoch"]
        mse = epoch_entry["mse"]
        pattern_logs = epoch_entry["pattern_logs"]

        lines.append(r"\subsubsection*{" + rf"Época {epoch}" + r"}")
        lines.append(
            rf"\textit{{Error cuadrático medio}}: "
            rf"$\text{{MSE}} = {_fmt_number(mse)}$."
        )

        # Tabla principal
        lines.append(r"\begin{tabular}{c c c c c c c c}")
        lines.append(r"\toprule")
        lines.append(
            r"Patrón & $\mathbf{x}$ & $y$ & "
            r"$\text{net}_o$ & $\hat{y}$ & $e$ & "
            r"$\delta_o$ & MSE parcial \\"
        )
        lines.append(r"\midrule")

        for idx, log in enumerate(pattern_logs, start=1):
            x_vec = log["x"]
            y_tgt = log["y_target"]
            net_o = log["net_o"]
            y_hat = log["y_hat"]
            e = log["error"]
            delta_o = log["delta_o"]

            lines.append(
                rf"{idx} & $({_fmt_vector(x_vec)})$ & {y_tgt} & "
                rf"{_fmt_number(net_o)} & {_fmt_number(y_hat)} & "
                rf"{_fmt_number(e)} & {_fmt_number(delta_o)} & "
                rf"{_fmt_number(e*e)} \\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\\[1em]")

        # Pesos resumen
        lines.append(r"\textit{Pesos después de esta época:}")
        lines.append(
            rf"\[ W_{{\text{{oculta}}}} = {_fmt_matrix(epoch_entry['weights_hidden_after'])} \]"
        )
        lines.append(
            rf"\[ W_{{\text{{salida}}}} = {_fmt_vector(epoch_entry['weights_output_after'])} \]"
        )
        lines.append(r"\\[1em]")

    # ========================================================
    # 5. Conclusión
    # ========================================================
    lines.append(r"\subsection*{Conclusiones del entrenamiento Backpropagation}")

    if final_mse <= error_threshold:
        lines.append(
            r"El MLP logra aproximar la función objetivo reduciendo el error "
            r"por debajo del umbral especificado, evidenciando que el problema "
            r"es no linealmente separable pero sí aproximable con una capa "
            r"oculta adecuada."
        )
    else:
        lines.append(
            r"El entrenamiento no alcanzó el umbral de error; esto puede indicar "
            r"que el número de neuronas ocultas es insuficiente, la tasa de "
            r"aprendizaje no es adecuada o que el problema requiere múltiples "
            r"capas o funciones de activación distintas."
        )

    lines.append(r"\newpage")

    return "\n".join(lines)

