#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/report/latex_backprop.py
------------------------------------------------------------
Bloque LaTeX para Backpropagation (MLP con 1 capa oculta).
------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from src.report.report_latex import escape_latex, dataframe_to_latex_table


def _fmt_matrix(M):
    if M.ndim == 1:
        return " \\\\ ".join(f"{v:.4f}" for v in M)

    rows = []
    for row in M:
        rows.append(" & ".join(f"{v:.4f}" for v in row) + " \\\\")
    return "\n".join(rows)


def build_backprop_block(result, df, X_cols, Y_col,
                         lr, hidden_neurons, max_epochs):

    W_hidden_final = result.final_hidden_weights
    W_output_final = result.final_output_weights
    mse_history = result.mse_history

    converged = result.converged
    convergence_epoch = result.convergence_epoch

    df_preview = dataframe_to_latex_table(
        df,
        caption=f"Vista previa del dataset (Backprop — {escape_latex(Y_col)})"
    )

    mse_fmt = "\n".join(
        f"Epoch {i}: MSE = {m:.6f}"
        for i, m in enumerate(mse_history)
    )

    hidden_w_hist = []
    for i, W in enumerate(result.hidden_weights_history):
        m = ", ".join("[" + ", ".join(f"{v:.4f}" for v in row) + "]" for row in W)
        hidden_w_hist.append(f"Epoch {i}: {m}")
    hidden_w_hist_fmt = "\n".join(hidden_w_hist)

    output_w_hist = []
    for i, W in enumerate(result.output_weights_history):
        m = ", ".join(f"{v:.4f}" for v in W.flatten())
        output_w_hist.append(f"Epoch {i}: [{m}]")
    output_w_hist_fmt = "\n".join(output_w_hist)

    conv_text = (
        f"La red convergió en la época {convergence_epoch}."
        if converged else
        "La red **no** convergió dentro del número máximo de épocas."
    )

    latex = f"""
% ============================================================
\\section*{{Backpropagation (MLP) — Tabla {escape_latex(Y_col)} }}
% ============================================================

\\subsection*{{Arquitectura de la red}}
\\begin{{itemize}}
  \\item Entradas: {len(X_cols)}
  \\item Neuronas ocultas: {hidden_neurons}
  \\item Neurona de salida: 1
  \\item Tasa de aprendizaje: $\\eta = {lr}$
  \\item Épocas máximas: {max_epochs}
\\end{{itemize}}

\\subsection*{{Dataset utilizado}}
{df_preview}

\\subsection*{{Pesos finales}}

\\[
W_{{hidden}} =
\\begin{{bmatrix}}
{_fmt_matrix(W_hidden_final)}
\\end{{bmatrix}}
\\]

\\[
W_{{output}} =
\\begin{{bmatrix}}
{_fmt_matrix(W_output_final)}
\\end{{bmatrix}}
\\]

\\subsection*{{Estado de convergencia}}
{conv_text}

\\subsection*{{Historial del Error (MSE por época)}}
\\begin{{Verbatim}}[breaklines=true, fontsize=\\scriptsize]
{mse_fmt}
\\end{{Verbatim}}

\\subsection*{{Historial de pesos — Capa oculta}}
\\begin{{Verbatim}}[breaklines=true, fontsize=\\scriptsize]
{hidden_w_hist_fmt}
\\end{{Verbatim}}

\\subsection*{{Historial de pesos — Capa de salida}}
\\begin{{Verbatim}}[breaklines=true, fontsize=\\scriptsize]
{output_w_hist_fmt}
\\end{{Verbatim}}

\\vspace{{0.6cm}}
"""

    return latex

