#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/report/latex_delta.py
------------------------------------------------------------
Descripción:
Generador del bloque LaTeX correspondiente a la Regla Delta
(Widrow–Hoff / ADALINE), siguiendo EXCLUSIVAMENTE el método
presentado en la práctica de Redes Neuronales.

Incluye:
  - Modelo lineal (neurona ADALINE)
  - Salida continua: ŷ = net
  - Error lineal: e = y - ŷ
  - Regla de actualización:
        Δw_i = η * e * x_i
  - Cálculo del Error Cuadrático Medio (MSE)
  - Tablas paso a paso: net, ŷ, error, Δw, pesos antes/después

Este módulo NO agrega conceptos ajenos:
  - No usa función escalón.
  - No usa derivadas (excepto las triviales del modelo lineal).
  - No usa matrices ni optimizaciones adicionales.

Consumido por:
  - src/report/report_nn_builder.py

Requiere:
  - src/nn/delta_rule.py (DeltaRuleResult)
  - src/report/report_latex.py (_fmt_number, dataset_preview_table)
------------------------------------------------------------
"""

from __future__ import annotations
from typing import List
import pandas as pd

from src.nn.delta_rule import DeltaRuleResult
from src.report.report_latex import _fmt_number, dataset_preview_table


# ============================================================
# === Utilidades internas ====================================
# ============================================================

def _fmt_vector(v: List[float]) -> str:
    """Formato simple tipo (1.0, 2.0, 3.0) para LaTeX."""
    return "(" + ", ".join(_fmt_number(x) for x in v) + ")"


# ============================================================
# === Bloque completo de Regla Delta ==========================
# ============================================================

def build_delta_rule_block(
    df_num: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    learning_rate: float,
    max_epochs: int,
    error_threshold: float,
    result: DeltaRuleResult,
    section_title: str | None = None,
) -> str:
    """
    Construye el bloque LaTeX completo para el experimento con
    Regla Delta (ADALINE), siguiendo el procedimiento del PDF.

    Parámetros
    ----------
    df_num : pd.DataFrame
        Dataset numérico limpiado.
    x_cols : List[str]
        Columnas de entrada.
    y_col : str
        Nombre de la columna objetivo.
    learning_rate : float
        Tasa de aprendizaje usada.
    max_epochs : int
        Épocas máximas configuradas.
    error_threshold : float
        Umbral de MSE para criterio de paro.
    result : DeltaRuleResult
        Resultados del entrenamiento.
    section_title : str | None
        Título personalizado del bloque.

    Retorna
    -------
    str
        Bloque LaTeX completo listo para insertarse.
    """
    lines: List[str] = []

    if section_title is None:
        section_title = (
            f"Regla Delta para la función {y_col} "
            f"con entradas ({', '.join(x_cols)})"
        )

    # ========================================================
    # 1. Título y resumen del dataset
    # ========================================================
    lines.append(f"\\section*{{{section_title}}}")
    lines.append(
        r"En esta sección se documenta el entrenamiento de una neurona "
        r"lineal (ADALINE) mediante la \textbf{Regla Delta}, tal como "
        r"se expone en la presentación del curso."
    )

    lines.append(r"\subsection*{Resumen del dataset}")
    lines.append(dataset_preview_table(df_num))

    # ========================================================
    # 2. Modelo ADALINE y regla Delta
    # ========================================================
    lines.append(r"\subsection*{Modelo ADALINE y Regla Delta}")

    lines.append(r"\textbf{Neurona lineal:}")
    lines.append(
        r"\["
        r"\text{net} = \sum_{i=1}^{n} w_i x_i"
        r"\]"
    )
    lines.append(
        r"La salida del modelo es simplemente: "
        r"\[ \hat{y} = \text{net} \]"
    )

    lines.append(r"\textbf{Error lineal:}")
    lines.append(r"\[ e = y - \hat{y} \]")

    lines.append(r"\textbf{Regla de actualización (Widrow–Hoff):}")
    lines.append(
        r"\["
        r"w_i^{(\text{nuevo})} = w_i^{(\text{viejo})} + \eta \, e \, x_i"
        r"\]"
    )

    lines.append(
        r"Se minimiza el error cuadrático medio (MSE):"
        r"\["
        r"\text{MSE} = \frac{1}{N}\sum_{p=1}^{N}(y_p - \hat{y}_p)^2"
        r"\]"
    )

    # ========================================================
    # 3. Parámetros del entrenamiento
    # ========================================================
    lines.append(r"\subsection*{Parámetros de entrenamiento}")
    lines.append(
        r"\begin{itemize}"
        rf"\item Tasa de aprendizaje: $\eta = {_fmt_number(learning_rate)}$."
        rf"\item Máximo de épocas: $N_{{\text{{max}}}} = {max_epochs}$."
        rf"\item Umbral de paro MSE: $\epsilon = {_fmt_number(error_threshold)}$."
        rf"\item Pesos iniciales: ${_fmt_vector(result.weights_initial)}$."
        r"\end{itemize}"
    )

    # ========================================================
    # 4. Desarrollo del entrenamiento
    # ========================================================
    lines.append(r"\subsection*{Desarrollo del entrenamiento por épocas}")

    # Mostrar si se alcanzó el umbral
    final_mse = result.mse_history[-1]
    if final_mse <= error_threshold:
        lines.append(
            rf"La red alcanza el umbral de error en la época "
            rf"$E = {result.epochs}$ "
            rf"con un MSE final de ${_fmt_number(final_mse)}$."
        )
    else:
        lines.append(
            rf"La red \textbf{no alcanza} el umbral de error "
            rf"en las $N_{{\text{{max}}}} = {max_epochs}$ épocas configuradas. "
            r"A continuación se muestran las tablas detalladas."
        )

    # Épocas
    for epoch_entry in result.epoch_history:
        epoch = epoch_entry["epoch"]
        mse = epoch_entry["mse"]
        pattern_logs = epoch_entry["pattern_logs"]

        lines.append(r"\subsubsection*{" + rf"Época {epoch}" + r"}")
        lines.append(
            rf"\textit{{Error cuadrático medio (MSE):}} "
            rf"${_fmt_number(mse)}$."
        )

        # Tabla principal de la época
        lines.append(r"\begin{tabular}{c c c c c c c}")
        lines.append(r"\toprule")
        lines.append(
            r"Patrón & $\mathbf{x}$ & $y$ & $\text{net}$ & "
            r"$\hat{y}$ & $e$ & $(\Delta w_1, \dots, \Delta w_n)$ \\"
        )
        lines.append(r"\midrule")

        for idx, plog in enumerate(pattern_logs, start=1):
            x_vec = plog["x"]
            y_target = plog["y_target"]
            net = plog["net"]
            y_hat = plog["y_hat"]
            error = plog["error"]
            delta_w = plog["delta_w"]

            lines.append(
                rf"{idx} & $({_fmt_vector(x_vec)})$ & {y_target} & "
                rf"{_fmt_number(net)} & {_fmt_number(y_hat)} & "
                rf"{_fmt_number(error)} & $({_fmt_vector(delta_w)})$ \\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\\[1em]")

        # salto explicativo
        w_before = _fmt_vector(pattern_logs[-1]["w_before"])
        w_after  = _fmt_vector(pattern_logs[-1]["w_after"])
        lines.append(
            r"\textit{Resumen de la época " + f"{epoch}" + r":} "
            r"Los pesos pasan de "
            rf"$\mathbf{{w}}^\text{{antes}} = {w_before}$ "
            r"a "
            rf"$\mathbf{{w}}^\text{{después}} = {w_after}$."
        )
        lines.append(r"\\[1em]")

    # ========================================================
    # 5. Conclusiones
    # ========================================================
    lines.append(r"\subsection*{Conclusiones del entrenamiento con Regla Delta}")

    if final_mse <= error_threshold:
        lines.append(
            r"La neurona lineal logra ajustar sus pesos de forma que el "
            r"error cuadrático medio cae por debajo del umbral especificado. "
            r"Esto muestra que la función objetivo es linealmente aproximable "
            r"por un modelo ADALINE."
        )
    else:
        lines.append(
            r"El entrenamiento no alcanza el umbral requerido. Esto puede "
            r"indicar que la función objetivo no es linealmente aproximable "
            r"por una sola neurona ADALINE (como ocurre en el caso de XOR)."
        )

    lines.append(r"\newpage")

    return "\n".join(lines)

