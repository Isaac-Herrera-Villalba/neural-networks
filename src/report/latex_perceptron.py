#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/report/latex_perceptron.py
------------------------------------------------------------
Generador del bloque LaTeX para Perceptrón Simple.
Compatible 100% con PerceptronResult.
------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from src.report.report_latex import escape_latex, dataframe_to_latex_table


def build_perceptron_block(result, df, X_cols, Y_col,
                           lr, threshold, initial_weights, max_epochs):

    # ------------------------------------------------------------
    # Datos del modelo
    # ------------------------------------------------------------
    W_final = result.final_weights
    W_history = result.weights_history
    error_history = result.error_history
    convergence_epoch = result.convergence_epoch
    converged = result.converged

    # ------------------------------------------------------------
    # Formateo
    # ------------------------------------------------------------
    w_final_fmt = ",\\, ".join(f"{v:.3f}" for v in W_final)

    # Historial de pesos por época
    w_hist_lines = []
    for i, w in enumerate(W_history):
        w_str = ", ".join(f"{v:.3f}" for v in w)
        w_hist_lines.append(f"Epoch {i}: [{w_str}]")
    w_hist_fmt = "\n".join(w_hist_lines)

    # Tabla previa del dataset
    df_preview = dataframe_to_latex_table(
        df,
        caption=f"Vista previa del dataset ({escape_latex(Y_col)})"
    )

    # Historial de errores
    err_fmt = "\n".join(f"Epoch {i}: errores = {e}" for i, e in enumerate(error_history))

    # Convergencia
    conv_text = (
        f"La red convergió en la época {convergence_epoch}."
        if converged else
        "La red **no** convergió."
    )

    # ------------------------------------------------------------
    # Construcción del bloque LaTeX
    # ------------------------------------------------------------
    latex = f"""
% ------------------------------------------------------------
\\section*{{Perceptrón — Tabla {escape_latex(Y_col)} }}
% ------------------------------------------------------------

\\subsection*{{Parámetros del experimento}}
\\begin{{itemize}}
  \\item Tasa de aprendizaje: $\\eta = {lr}$
  \\item Umbral (bias): $b = {threshold}$
  \\item Pesos iniciales: {escape_latex(str(initial_weights) if initial_weights is not None else "Inicialización aleatoria")}
  \\item Máximo de épocas: {max_epochs}
\\end{{itemize}}

\\subsection*{{Dataset utilizado}}
{df_preview}

\\subsection*{{Resultados finales}}
\\begin{{itemize}}
  \\item Pesos finales: $W = [{w_final_fmt}]$
  \\item Convergencia: {conv_text}
\\end{{itemize}}

\\subsection*{{Historial de pesos por época}}
\\begin{{verbatim}}
{w_hist_fmt}
\\end{{verbatim}}

\\subsection*{{Historial de errores por época}}
\\begin{{verbatim}}
{err_fmt}
\\end{{verbatim}}

\\vspace{{0.6cm}}
"""
    return latex

