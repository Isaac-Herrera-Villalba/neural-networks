#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/report/latex_perceptron.py
-------------------------------------------------------------------------------
Generador del bloque LaTeX para el Perceptrón Simple de Rosenblatt
-------------------------------------------------------------------------------

Descripción general
-------------------
Este módulo construye dinámicamente el bloque LaTeX correspondiente al análisis
de entrenamiento del perceptrón simple, incorporando:

    • La definición formal del perceptrón (como clasificador lineal).
    • La regla de aprendizaje Δw_i = η(t − o)x_i.
    • El algoritmo de entrenamiento en pseudocódigo.
    • La ecuación del hiperplano aprendido.
    • Historial completo de pesos {w^(epoch)}.
    • Historial de errores por época.
    • Análisis de convergencia / no convergencia.
    • Interpretación conceptual basada en las diapositivas 1–8 de
      “Learning Perceptrons”.

El objetivo de este módulo es producir un bloque LaTeX autocontenido,
académicamente correcto y listo para su inclusión en `reporte_nn.tex`.

Este módulo es utilizado únicamente por:
    - src/report/report_nn_builder.py
-------------------------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from src.report.report_latex import escape_latex, dataframe_to_latex_table


def build_perceptron_block(result, df, X_cols, Y_col,
                           lr, threshold, initial_weights, max_epochs):
    """
    Construye el bloque LaTeX completo asociado al entrenamiento de un perceptrón.

    Parámetros
    ----------
    result : PerceptronResult
        Objeto que contiene pesos, errores y estado final del entrenamiento.

    df : pd.DataFrame
        Dataset original cargado desde .ods.

    X_cols : list[str]
        Nombres lógicos de las columnas de entrada.

    Y_col : str
        Nombre de la columna objetivo.

    lr : float
        Tasa de aprendizaje η.

    threshold : float
        Sesgo adicional fijo sumado a la activación. Normalmente 0.

    initial_weights : np.ndarray | None
        Vector de pesos iniciales proporcionado por el usuario o None.

    max_epochs : int
        Límite superior de épocas de entrenamiento.

    Retorna
    -------
    str
        Código LaTeX del bloque, listo para ser incluido en el documento final.
    """

    # -------------------------------------------------------------------------
    # Extracción directa de resultados
    # -------------------------------------------------------------------------
    W_final = result.final_weights
    W_history = result.weights_history
    error_history = result.error_history
    convergence_epoch = result.convergence_epoch
    converged = result.converged

    # -------------------------------------------------------------------------
    # Construcción simbólica del hiperplano: w0 + w1 x1 + ... + wn xn
    # -------------------------------------------------------------------------
    terms = [f"{W_final[0]:.4f}"]
    for i, v in enumerate(W_final[1:], start=1):
        terms.append(f"{v:.4f} x_{i}")
    hyperplane = " + ".join(terms)

    # -------------------------------------------------------------------------
    # Historial de pesos por época
    # -------------------------------------------------------------------------
    w_hist_lines = []
    for i, w in enumerate(W_history):
        w_str = ", ".join(f"{v:.4f}" for v in w)
        w_hist_lines.append(f"Epoch {i}: [{w_str}]")
    w_hist_fmt = "\n".join(w_hist_lines)

    # -------------------------------------------------------------------------
    # Diferencias entre pesos sucesivos (Δw_i por época)
    # -------------------------------------------------------------------------
    delta_w_lines = []
    if len(W_history) >= 2:
        for epoch_idx in range(len(W_history) - 1):
            w_prev = W_history[epoch_idx]
            w_next = W_history[epoch_idx + 1]
            dw = w_next - w_prev
            dws = ", ".join(f"Δw_{i} = {dw[i]:.4f}" for i in range(len(dw)))
            delta_w_lines.append(f"Epoch {epoch_idx} → {epoch_idx + 1}: {dws}")
    else:
        delta_w_lines.append(
            "Solo se registró una época; no hay diferencias entre épocas sucesivas."
        )
    delta_w_fmt = "\n".join(delta_w_lines)

    # -------------------------------------------------------------------------
    # Historial de errores
    # -------------------------------------------------------------------------
    err_fmt = "\n".join(f"Epoch {i}: errores = {e}" for i, e in enumerate(error_history))

    # -------------------------------------------------------------------------
    # Explicación de convergencia o no convergencia
    # -------------------------------------------------------------------------
    if converged:
        convergence_explanation = (
            f"La red convergió en la época {convergence_epoch}. "
            r"Esto indica que el perceptrón encontró un conjunto de pesos que "
            r"clasifica correctamente todos los patrones del dataset. "
            r"Este comportamiento coincide con la teoría: el perceptrón converge "
            r"siempre que el conjunto de entrenamiento sea \textit{linealmente separable}."
        )

        conclusion_text = (
            r"El experimento confirma que la función objetivo puede representarse "
            r"con un hiperplano que divide el espacio de entrada en dos regiones. "
            r"El perceptrón simple es suficiente para resolver este problema."
        )
    else:
        convergence_explanation = (
            r"La red \textbf{no convergió}. Esto ocurre típicamente cuando los datos "
            r"no son \textit{linealmente separables}, es decir, no existe ningún "
            r"hiperplano que divida perfectamente las clases. "
            r"Casos clásicos como XOR ilustran esta limitación del perceptrón simple."
        )

        conclusion_text = (
            r"Debido a la no separabilidad lineal del problema, el perceptrón simple "
            r"no puede representar la frontera de decisión requerida. "
            r"Para este tipo de problemas se debe emplear una arquitectura más potente, "
            r"como un perceptrón multicapa (MLP)."
        )

    # -------------------------------------------------------------------------
    # Vista previa del dataset
    # -------------------------------------------------------------------------
    df_preview = dataframe_to_latex_table(
        df,
        caption=f"Vista previa del dataset ({escape_latex(Y_col)})"
    )

    # -------------------------------------------------------------------------
    # Ensamble final del bloque LaTeX
    # -------------------------------------------------------------------------
    latex = f"""
% ------------------------------------------------------------
\\section*{{Perceptrón — Tabla {escape_latex(Y_col)} }}
% ------------------------------------------------------------

\\subsection*{{1. Definición del perceptrón}}

Un perceptrón es un clasificador lineal definido por la regla:

\\[
o(x_1, x_2, \\ldots, x_n) =
\\begin{{cases}}
1 & \\text{{si }} w_0 + w_1 x_1 + \\cdots + w_n x_n > 0 \\\\
-1 & \\text{{en otro caso}}
\\end{{cases}}
\\]

\\subsection*{{2. Regla de aprendizaje}}

\\[
w_i = w_i + \\Delta w_i \dots (1)
\\]

\\[
\\Delta w_i = \\eta (t - o)x_i \dots (2)
\\]

Donde:
\\begin{{itemize}}
    \\item $\\eta$ es la tasa de aprendizaje.
    \\item $t$ es la clasificación objetivo.
    \\item $o$ es la clasificación actual.
\\end{{itemize}}

\\subsection*{{3. Algoritmo de entrenamiento}}
\\begin{{enumerate}}
    \\item Inicializar pesos.
    \\item Para cada época:
    \\begin{{enumerate}}
        \\item Para cada patrón: calcular salida, comparar con etiqueta real.
        \\item Si hay error, actualizar pesos con la regla anterior.
    \\end{{enumerate}}
    \\item Si en una época no hay errores, el modelo converge.
\\end{{enumerate}}

\\subsection*{{4. Parámetros del experimento}}
\\begin{{itemize}}
  \\item Tasa de aprendizaje: $\\eta = {lr}$
  \\item Pesos iniciales: {escape_latex(str(initial_weights) if initial_weights else "Inicialización aleatoria")}
  \\item Máximo de épocas: {max_epochs}
\\end{{itemize}}

\\subsection*{{5. Dataset utilizado}}
{df_preview}

\\subsection*{{6. Resultados finales}}
\\begin{{itemize}}
  \\item Pesos finales: $W = [{",\\, ".join(f"{v:.4f}" for v in W_final)}]$
  \\item Hiperplano aprendido: \\[
     {hyperplane} = 0
  \\]
  \\item Estado de convergencia: {convergence_explanation}
\\end{{itemize}}

\\subsection*{{7. Historial de pesos}}
\\begin{{Verbatim}}[breaklines=true, fontsize=\\scriptsize]
{w_hist_fmt}
\\end{{Verbatim}}

\\subsection*{{8. Aproximación de $\\Delta w_i$ por época}}
\\begin{{Verbatim}}[breaklines=true, fontsize=\\scriptsize]
{delta_w_fmt}
\\end{{Verbatim}}

\\subsection*{{9. Historial de errores}}
\\begin{{Verbatim}}[breaklines=true, fontsize=\\scriptsize]
{err_fmt}
\\end{{Verbatim}}

\\subsection*{{10. Conclusiones}}
{conclusion_text}

\\vspace{{0.6cm}}
"""
    return latex

