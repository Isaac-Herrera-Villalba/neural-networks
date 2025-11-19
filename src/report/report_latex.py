#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/report/report_latex.py
------------------------------------------------------------
Funciones auxiliares para:
 - Escapar caracteres LaTeX
 - Convertir DataFrames a tablas LaTeX
 - Generar documentos PDF completos desde bloques de LaTeX
------------------------------------------------------------
"""

from __future__ import annotations
import subprocess
from pathlib import Path
import pandas as pd


# ============================================================
# ESCAPAR CARACTERES ESPECIALES DE LATEX
# ============================================================

def escape_latex(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for a, b in repl.items():
        text = text.replace(a, b)
    return text


# ============================================================
# CONVERTIR DATAFRAME A TABLA LATEX
# ============================================================


def dataframe_to_latex_table(df: pd.DataFrame, caption: str | None = None, max_rows: int = 20) -> str:
    """
    Convierte un DataFrame en una tabla LaTeX sin usar pandas.to_latex()
    para evitar dependencia con Jinja2.
    """

    if df.empty:
        return r"\textbf{(Dataset vacío)}"

    df2 = df.head(max_rows).copy()
    df2 = df2.applymap(escape_latex)

    # -------- Construcción manual de tabla LaTeX --------
    cols = list(df2.columns)
    header = " & ".join(cols) + r" \\ \hline"

    rows = []
    for _, row in df2.iterrows():
        rows.append(" & ".join(str(v) for v in row) + r" \\")
    body = "\n".join(rows)

    table_latex = (
        r"\begin{table}[H]" "\n"
        r"\centering" "\n"
        r"\begin{tabular}{|" + "c|"*len(cols) + "}" "\n"
        r"\hline" "\n"
        + header + "\n"
        + body + "\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
    )

    if caption:
        table_latex += r"\caption{" + escape_latex(caption) + "}\n"

    table_latex += r"\end{table}" "\n"

    note = (
        r"\newline{\small (Sólo se muestran las primeras "
        + f"{max_rows}"
        + r" filas.)}"
    )

    return table_latex + "\n" + note + "\n"


# ============================================================
# RENDERIZAR UN PDF COMPLETO
# ============================================================

def render_all_instances_pdf(output_pdf_path: str, latex_body: str):
    """
    Construye un documento LaTeX completo, lo compila con pdflatex
    y genera el PDF en output/.
    """

    output_pdf_path = Path(output_pdf_path)
    output_dir = output_pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    tex_path = output_dir / output_pdf_path.with_suffix(".tex").name

    # Documento LaTeX completo
    latex_full = r"""
\documentclass[12pt]{article}
\usepackage[spanish]{babel}
\usepackage{amsmath, amssymb}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{float}

\begin{document}
""" + latex_body + r"""
\end{document}
"""

    # Guardar .tex
    tex_path.write_text(latex_full, encoding="utf-8")

    # Compilar PDF
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        tex_path.name
    ]

    # Ejecutar pdflatex dentro del output/
    subprocess.run(cmd, cwd=output_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return str(output_pdf_path)

