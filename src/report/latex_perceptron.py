#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/report/latex_perceptron.py
------------------------------------------------------------
Descripción:
Generador del bloque LaTeX correspondiente al entrenamiento de
un Perceptrón simple, siguiendo EXCLUSIVAMENTE el procedimiento
mostrado en la presentación de Redes Neuronales.

Este módulo NO introduce matrices ni formulaciones distintas;
se limita a:

  - Describir el modelo del perceptrón:
        net = Σ w_i x_i
        ŷ = f(net) (función escalón)
  - Mostrar la regla de aprendizaje:
        w_i^{nuevo} = w_i^{viejo} + η e x_i
        e = y - ŷ
  - Documentar paso a paso, por época y por patrón:
        - net
        - salida ŷ
        - error e
        - Δw
        - pesos antes y después

Estructura del bloque LaTeX generado:
------------------------------------------------------------
  1. Encabezado de sección.
  2. Resumen del dataset (tabla).
  3. Modelo del perceptrón y regla de actualización.
  4. Parámetros de entrenamiento (η, épocas, pesos iniciales).
  5. Desarrollo del entrenamiento:
       - una subsección por época
       - tabla de patrones por época.

Este módulo es consumido por:
  - src/report/report_nn_builder.py

Requiere:
  - src/nn/perceptron.py (PerceptronResult)
  - src/report/report_latex.py (_fmt_number, dataset_preview_table)
------------------------------------------------------------
"""

from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

from src.nn.perceptron import PerceptronResult
from src.report.report_latex import _fmt_number, dataset_preview_table


# ============================================================
# === UTILIDADES INTERNAS ====================================
# ============================================================

def _fmt_vector(vec: List[float]) -> str:
    """
    Formatea una lista de valores como vector entre paréntesis
    en notación LaTeX amigable:
        (v1, v2, v3)

    Parámetros
    ----------
    vec : List[float]

    Retorna
    -------
    str
        Cadena preparada para incrustar en texto LaTeX.
    """
    return "(" + ", ".join(_fmt_number(v) for v in vec) + ")"


# ============================================================
# === BLOQUE PRINCIPAL DE PERCEPTRÓN =========================
# ============================================================

def build_perceptron_block(
    df_num: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    learning_rate: float,
    max_epochs: int,
    result: PerceptronResult,
    section_title: str | None = None,
) -> str:
    """
    Construye el bloque LaTeX completo para un experimento con
    Perceptrón simple.

    Parámetros
    ----------
    df_num : pd.DataFrame
        Dataset numérico ya preprocesado.
    x_cols : List[str]
        Nombres de las columnas de entrada (X).
    y_col : str
        Nombre de la columna objetivo (Y).
    learning_rate : float
        Tasa de aprendizaje η utilizada.
    max_epochs : int
        Número máximo de épocas configurado.
    result : PerceptronResult
        Resultado del entrenamiento devuelto por train_perceptron().
    section_title : str | None
        Título personalizado de la sección; si es None, se genera
        uno genérico basado en Y y X.

    Retorna
    -------
    str
        Cadena con el bloque LaTeX listo para insertarse en el
        documento principal.
    """
    lines: List[str] = []

    if section_title is None:
        section_title = (
            f"Perceptrón para la función {y_col} "
            f"con entradas ({', '.join(x_cols)})"
        )

    # ========================================================
    # 1. Encabezado y resumen del dataset
    # ========================================================
    lines.append(f"\\section*{{{section_title}}}")
    lines.append(
        r"En esta sección se detalla el procedimiento de entrenamiento "
        r"de un perceptrón simple siguiendo el algoritmo visto en clase."
    )

    lines.append(r"\subsection*{Resumen del dataset}")
    lines.append(dataset_preview_table(df_num))

    # ========================================================
    # 2. Modelo del Perceptrón y regla de aprendizaje
    # ========================================================
    lines.append(r"\subsection*{Modelo del Perceptrón y regla de aprendizaje}")

    # Modelo
    lines.append(r"\textbf{Modelo de neurona:}")
    lines.append(
        r"\["
        r"\text{net} = \sum_{i=1}^{n} w_i x_i"
        r"\]"
    )
    lines.append(
        r"Donde $x_i$ son las entradas, $w_i$ los pesos sinápticos y "
        r"$\text{net}$ es la combinación lineal de ambas."
    )

    # Función escalón
    lines.append(r"\textbf{Función de activación (escalón):}")
    lines.append(
        r"\["
        r"\hat{y} = f(\text{net}) = "
        r"\begin{cases}"
        r"1, & \text{si } \text{net} \ge 0,\\"
        r"0, & \text{si } \text{net} < 0."
        r"\end{cases}"
        r"\]"
    )

    # Regla de actualización
    lines.append(r"\textbf{Regla de aprendizaje del Perceptrón:}")
    lines.append(
        r"\["
        r"e = y - \hat{y}"
        r"\]"
    )
    lines.append(
        r"\["
        r"w_i^{(\text{nuevo})} = w_i^{(\text{viejo})} + \eta \, e \, x_i"
        r"\]"
    )
    lines.append(
        r"Donde $\eta$ es la tasa de aprendizaje, $e$ el error "
        r"de clasificación y $x_i$ la $i$-ésima entrada del patrón."
    )

    # ========================================================
    # 3. Parámetros de entrenamiento
    # ========================================================
    lines.append(r"\subsection*{Parámetros de entrenamiento}")
    lines.append(
        r"\begin{itemize}"
        rf"\item Tasa de aprendizaje: $\eta = {_fmt_number(learning_rate)}$."
        rf"\item Número máximo de épocas: $N_{{\text{{max}}}} = {max_epochs}$."
        rf"\item Pesos iniciales: ${_fmt_vector(result.weights_initial)}$."
        r"\end{itemize}"
    )

    # ========================================================
    # 4. Desarrollo del entrenamiento
    # ========================================================
    lines.append(r"\subsection*{Desarrollo del entrenamiento por épocas}")

    n_epochs = result.epochs

    if result.converged:
        lines.append(
            rf"El perceptrón \textbf{converge} en la época "
            rf"$E = {n_epochs}$, es decir, a partir de esa época "
            r"todos los patrones son clasificados correctamente."
        )
    else:
        lines.append(
            rf"El perceptrón \textbf{no convergió} en el número máximo "
            rf"de épocas configurado ($N_{{\text{{max}}}} = {max_epochs}$). "
            r"Se muestran a continuación las trazas de actualización."
        )

    # Una subsección por época
    for epoch_entry in result.epoch_history:
        epoch = epoch_entry["epoch"]
        pattern_logs = epoch_entry["pattern_logs"]

        lines.append(r"\subsubsection*{" + rf"Época {epoch}" + r"}")

        # Tabla por época
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
            y_pred = plog["y_pred"]
            err = plog["error"]
            delta_w = plog["delta_w"]

            x_str = _fmt_vector(x_vec)
            dw_str = _fmt_vector(delta_w)

            lines.append(
                rf"{idx} & ${x_str}$ & {y_target} & "
                rf"{_fmt_number(net)} & {y_pred} & {err} & ${dw_str}$ \\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\\[1em]")

        # Opcional: mostrar pesos antes/después (resumen)
        last_pattern = pattern_logs[-1]
        w_before = _fmt_vector(last_pattern["w_before"])
        w_after = _fmt_vector(last_pattern["w_after"])
        lines.append(
            r"\textit{Resumen de la época " + f"{epoch}" + r":} "
            r"los pesos pasan de "
            rf"$\mathbf{{w}}^\text{{antes}} = {w_before}$ "
            r"a "
            rf"$\mathbf{{w}}^\text{{después}} = {w_after}$."
        )
        lines.append(r"\\[1em]")

    # ========================================================
    # 5. Conclusión del bloque
    # ========================================================
    lines.append(r"\subsection*{Conclusiones del entrenamiento con Perceptrón}")

    if result.converged:
        lines.append(
            r"Se observa que el perceptrón logra encontrar un vector de pesos "
            r"que separa linealmente las clases del problema, cumpliendo con "
            r"el criterio de convergencia para el conjunto de patrones dado."
        )
    else:
        lines.append(
            r"A pesar de las iteraciones realizadas, el perceptrón no logra "
            r"encontrar un conjunto de pesos que clasifique correctamente "
            r"todos los patrones. Esto es consistente con la teoría, que indica "
            r"que ciertos problemas (por ejemplo, XOR) no son linealmente "
            r"separables y, por tanto, no pueden ser resueltos por un único "
            r"perceptrón."
        )

    lines.append(r"\newpage")

    return "\n".join(lines)

