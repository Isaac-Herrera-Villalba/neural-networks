#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/report/report_latex.py
------------------------------------------------------------
Descripción:
Módulo central para la generación del PDF final del proyecto
de Redes Neuronales. Aquí se definen:

1. Plantilla LaTeX base:
      - Preambulo moderno, con soporte matemático.
      - Márgenes adecuados y salida profesional.
      - Configuración para español.

2. Utilidades de conversión:
      - _fmt_number(): formateo numérico uniforme.
      - _matrix_to_latex(): matrices pequeñas para pesos.
      - dataset_preview_table(): tabla del dataset.

3. Función principal render_pdf():
      Toma un bloque LaTeX generado por:
        - latex_perceptron.py
        - latex_delta.py
        - latex_backprop.py
      y produce el PDF final.

Este módulo NO realiza cálculos de redes neuronales.
Solo formatea y renderiza.
------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import subprocess
import numpy as np
import pandas as pd


# ============================================================
# === PREÁMBULO LaTeX GENERAL ================================
# ============================================================

LATEX_PREAMBLE = r"""
\documentclass[11pt]{article}

\usepackage[margin=2.5cm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

\usepackage[spanish,es-noshorthands]{babel}
\usepackage{microtype}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{booktabs}

\usepackage{graphicx}
\usepackage{float}

\usepackage{breqn}  % ecuaciones largas

\usepackage{siunitx}
\sisetup{
  output-decimal-marker = {.},
  group-separator={\,},
  detect-all,
  locale = US
}

\usepackage{array}
\usepackage{ragged2e}
\usepackage{enumitem}

\setlength{\parskip}{0.8em}
\setlength{\parindent}{0pt}

\begin{document}
\RaggedRight
"""

LATEX_POSTAMBLE = r"""
\end{document}
"""


# ============================================================
# === UTILIDADES DE FORMATEO ================================
# ============================================================

def _fmt_number(x: float) -> str:
    """
    Formatea un número a 6 decimales, garantizando punto decimal.

    Ejemplo:
        0.5 -> "0.500000"
        1 -> "1.000000"
    """
    try:
        s = f"{float(x):.6f}"
    except Exception:
        s = str(x)
    return s.replace(",", ".")


def _matrix_to_latex(M: np.ndarray) -> str:
    """
    Convierte una matriz NumPy a entorno bmatrix.

    Se utiliza únicamente para pesos pequeños:
    - Pesos capa oculta
    - Pesos capa salida

    No se usa para cálculos matriciales grandes.
    """
    rows, cols = M.shape
    lines = []
    for i in range(rows):
        line = " & ".join(_fmt_number(M[i, j]) for j in range(cols))
        lines.append(line + r" \\")
    return "\\begin{bmatrix}\n" + "\n".join(lines) + "\n\\end{bmatrix}"


def dataset_preview_table(df: pd.DataFrame, max_rows: int = 12, max_cols: int = 8) -> str:
    """
    Genera tabla resumida del dataset en LaTeX.

    Ideal para mostrar las entradas y salidas del perceptrón, Regla Delta o Backprop.
    """
    total_r, total_c = df.shape
    df_disp = df.iloc[:max_rows, :max_cols].copy()

    # Escapar guiones bajos para LaTeX
    df_disp.columns = [str(c).replace("_", r"\_") for c in df_disp.columns]

    header_fmt = " ".join(["c"] * len(df_disp.columns))

    lines = []
    lines.append(
        rf"\textit{{Dimensiones del dataset:}} ${total_r}\,\text{{filas}} \times {total_c}\,\text{{columnas}}$"
    )
    lines.append("\\begin{tabular}{" + header_fmt + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(df_disp.columns) + r" \\")
    lines.append("\\midrule")

    for _, row in df_disp.iterrows():
        vals = [str(v).replace("_", r"\_") for v in row.values]
        lines.append(" & ".join(vals) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # mensaje si se truncó
    if total_r > max_rows or total_c > max_cols:
        lines.append(
            r"\\[0.3em]\textit{Nota: la tabla ha sido truncada para visualización. "
            r"Consulte el archivo original para ver todas las filas y columnas.}"
        )

    return "\n".join(lines)


# ============================================================
# === FUNCIÓN PRINCIPAL DE RENDER ============================
# ============================================================

def render_pdf(output_pdf: str, latex_body: str):
    """
    Renderiza un documento LaTeX ensamblado a un PDF final.

    Parámetros
    ----------
    output_pdf : str
        Ruta de salida del PDF.
    latex_body : str
        Contenido LaTeX generado por report_nn_builder.py

    Tareas:
      1. Ensambla preámbulo + cuerpo + cierre.
      2. Compila con pdflatex en modo silencioso.
      3. Produce el archivo final.

    Nota:
      - Requiere tener pdflatex instalado en el sistema.
      - Produce un archivo .tex junto al PDF para depuración.
    """
    out = Path(output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)

    tex_path = out.with_suffix(".tex")
    tex_data = LATEX_PREAMBLE + latex_body + LATEX_POSTAMBLE
    tex_path.write_text(tex_data, encoding="utf-8")

    # Ejecutar pdflatex dos veces
    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=out.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        if proc.returncode != 0:
            print("[ERROR] Falló la compilación LaTeX.")
            return

    print(f"[OK] PDF generado en: {out}")

