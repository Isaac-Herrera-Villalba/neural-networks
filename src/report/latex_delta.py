#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/report/latex_delta.py
------------------------------------------------------------
Bloque LaTeX para Regla Delta (ADALINE).

Compatible 100% con DeltaRuleResult:
    - weights_history
    - predictions_history
    - mse_history
    - final_weights
------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from src.report.report_latex import escape_latex, dataframe_to_latex_table


def build_delta_rule_block(result, df, X_cols, Y_col,
                           lr, initial_weights, max_epochs):

    # ============================================================
    #   Datos del resultado
    # ============================================================
    W_final = result.final_weights
    W_history = result.weights_history
    mse_history = result.mse_history
    preds_history = result.predictions_history

    converged = result.converged
    convergence_epoch = result.convergence_epoch

    # ============================================================
    #   Formateo de datos
    # ============================================================
    w_final_fmt = ",\\, ".join(f"{v:.4f}" for v in W_final)

    # Historial de pesos por época
    w_hist_lines = []
    for i, w in enumerate(W_history):
        w_str = ", ".join(f"{v:.4f}" for v in w)
        w_hist_lines.append(f"Epoch {i}: [{w_str}]")
    w_hist_fmt = "\n".join(w_hist_lines)

    # MSE por época
    mse_fmt = "\n".join(f"Epoch {i}: MSE = {m:.6f}" for i, m in enumerate(mse_history))

    # Predicciones continuas (útil para mostrar no-linealidad en XOR)
    pred_lines = []
    for i, pred in enumerate(preds_history):
        p_str = ", ".join(f"{v:.4f}" for v in pred)
        pred_lines.append(f"Epoch {i}: [{p_str}]")
    preds_fmt = "\n".join(pred_lines)

    # Tabla del dataset
    df_preview = dataframe_to_latex_table(
        df,
        caption=f"Vista previa del dataset (Regla Delta — {escape_latex(Y_col)})"
    )

    # Convergencia
    conv_text = (
        f"La red convergió en la época {convergence_epoch} (MSE por debajo del umbral)."
        if converged else
        "La red **no** convergió dentro del número de épocas."
    )

    # ============================================================
    #   Construcción LaTeX
    # ============================================================
    latex = f"""
% ------------------------------------------------------------
\\section*{{Regla Delta (ADALINE) — Tabla {escape_latex(Y_col)} }}
% ------------------------------------------------------------

\\subsection*{{Parámetros}}
\\begin{{itemize}}
  \\item Tasa de aprendizaje: $\\eta = {lr}$
  \\item Pesos iniciales: {escape_latex(str(initial_weights) if initial_weights is not None else "Inicialización aleatoria")}
  \\item Épocas máximas: {max_epochs}
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

\\subsection*{{Historial del Error Cuadrático Medio (MSE)}}
\\begin{{verbatim}}
{mse_fmt}
\\end{{verbatim}}

\\subsection*{{Predicciones continuas por época}}
\\begin{{verbatim}}
{preds_fmt}
\\end{{verbatim}}

\\vspace{{0.6cm}}
"""
    return latex

