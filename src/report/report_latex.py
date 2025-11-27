#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/report/report_latex.py
-------------------------------------------------------------------------------
Utilidades centrales para la generación de reportes LaTeX en el proyecto
Neural Networks (Perceptrón Simple)
-------------------------------------------------------------------------------

Descripción general
-------------------
Este módulo provee las funciones de soporte necesarias para construir el reporte
LaTeX final del proyecto. Su responsabilidad abarca tres tareas principales:

1. **escape_latex(text):**
      Escapa todos los caracteres que tienen un significado especial en LaTeX,
      garantizando que los contenidos provenientes del dataset, nombres de
      columnas o strings arbitrarios no generen errores de compilación.

2. **dataframe_to_latex_table(df, ...):**
      Convierte un DataFrame de pandas en una tabla LaTeX, sin depender de
      pandas.to_latex() (por compatibilidad, control manual y ausencia de Jinja2).
      Incluye un mecanismo para:
          - limitar filas y columnas mostradas,
          - informar cuántos datos fueron truncados,
          - producir tablas robustas con formato uniforme.

3. **render_all_instances_pdf(path, body):**
      Ensambla un documento LaTeX completo que incluye:
            - cabecera estándar,
            - paquetes necesarios (babel, amsmath, fancyvrb, graphicx, etc.),
            - contenido generado dinámicamente,
      lo escribe en `output/` y ejecuta `pdflatex` para producir el PDF.

Este módulo es utilizado exclusivamente por:
    - src/report/report_nn_builder.py
-------------------------------------------------------------------------------
"""

from __future__ import annotations
import subprocess
from pathlib import Path
import pandas as pd


# =============================================================================
# FUNCIÓN: escape_latex
# =============================================================================

def escape_latex(text: str) -> str:
    """
    Escapa caracteres especiales de LaTeX en una cadena.

    Esta función es necesaria porque muchos caracteres comunes
    (&, %, $, _, #, {, }, etc.) tienen significado sintáctico en LaTeX,
    y su presencia en texto normal provoca errores de compilación.

    Parámetros
    ----------
    text : str
        Cadena que debe ser escapada para ser insertada en LaTeX.

    Retorna
    -------
    str
        Cadena segura para incluir dentro de entornos LaTeX.

    Notas
    -----
    - Si el argumento no es str, se convierte a str automáticamente.
    - No agrega formateo adicional, únicamente escapa caracteres peligrosos.
    """
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


# =============================================================================
# FUNCIÓN: dataframe_to_latex_table
# =============================================================================

def dataframe_to_latex_table(
    df: pd.DataFrame,
    caption: str | None = None,
    max_rows: int = 20,
    max_cols: int = 20
) -> str:
    """
    Convierte un DataFrame en una tabla LaTeX con vista previa truncada.

    Este método reproduce manualmente la estructura tabular en LaTeX para
    garantizar total control sobre el formato, sin depender de pandas.to_latex.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset cargado desde archivo .ods/.xlsx/.csv.

    caption : str | None
        Título que acompaña la tabla en LaTeX.

    max_rows : int
        Número máximo de filas a mostrar en la vista previa.

    max_cols : int
        Número máximo de columnas a mostrar.

    Retorna
    -------
    str
        Código LaTeX de la tabla generada.

    Comportamiento
    --------------
    - Si el DataFrame excede max_rows o max_cols, se trunca.
    - Se agrega un mensaje informativo explicando cuánto se está mostrando
      respecto al total.
    - Se escapan caracteres especiales mediante escape_latex().
    """

    if df.empty:
        return r"\textbf{(Dataset vacío)}"

    total_rows = len(df)
    total_cols = len(df.columns)

    # -------- Recortar filas --------
    df2 = df.head(max_rows).copy()

    # -------- Recortar columnas --------
    if total_cols > max_cols:
        df2 = df2.iloc[:, :max_cols]

    # Escape de caracteres peligrosos para LaTeX
    df2 = df2.applymap(escape_latex)

    # Construcción manual de la tabla
    cols = list(df2.columns)
    header = " & ".join(cols) + r" \\ \hline"

    rows = []
    for _, row in df2.iterrows():
        rows.append(" & ".join(str(v) for v in row) + r" \\")
    body = "\n".join(rows)

    table_latex = (
        r"\begin{table}[H]" "\n"
        r"\centering" "\n"
        r"\begin{tabular}{|" + "c|" * len(cols) + "}" "\n"
        r"\hline" "\n"
        + header + "\n"
        + body + "\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
    )

    if caption:
        table_latex += r"\caption{" + escape_latex(caption) + "}\n"

    table_latex += r"\end{table}" "\n"

    # Mensaje de truncamiento (solo si aplica)
    trimmed_rows = (total_rows > max_rows)
    trimmed_cols = (total_cols > max_cols)

    if trimmed_rows or trimmed_cols:
        note = (
            r"\newline{\small (Solo se muestran "
            f"{min(max_rows, total_rows)} filas y "
            f"{min(max_cols, total_cols)} columnas de un total de "
            f"{total_rows} filas y {total_cols} columnas. "
            r"Consulta el dataset para revisar todos los datos.)}"
        )
        return table_latex + "\n" + note + "\n"

    return table_latex + "\n"


# =============================================================================
# FUNCIÓN: render_all_instances_pdf
# =============================================================================

def render_all_instances_pdf(output_pdf_path: str, latex_body: str):
    """
    Genera un documento PDF completo desde bloques de LaTeX.

    Esta función:
      1. Ensambla un documento LaTeX completo (cabecera + contenido).
      2. Lo escribe en output/ como .tex.
      3. Ejecuta pdflatex para producir el PDF final.

    Parámetros
    ----------
    output_pdf_path : str
        Ruta donde se generará el archivo PDF (debe estar en output/).

    latex_body : str
        Código LaTeX correspondiente al contenido generado para cada bloque NN.

    Retorna
    -------
    str
        Ruta final del archivo PDF.

    Notas
    -----
    • Se usa `pdflatex -interaction=nonstopmode` para evitar bloqueos por
      errores menores.
    • Todos los warnings o errores se descartan en stdout/stderr para que la
      salida de terminal del proyecto permanezca limpia.
    • El usuario puede recompilar manualmente via `make latex`.
    """

    output_pdf_path = Path(output_pdf_path)
    output_dir = output_pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    tex_path = output_dir / output_pdf_path.with_suffix(".tex").name

    # Cabecera completa del documento LaTeX
    latex_full = r"""
\documentclass[12pt]{article}
\usepackage[spanish]{babel}
\decimalpoint
\usepackage{amsmath, amssymb}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{float}
\usepackage{fancyvrb}
\usepackage{fvextra}

\begin{document}
""" + latex_body + r"""
\end{document}
"""

    tex_path.write_text(latex_full, encoding="utf-8")

    # Ejecutar pdflatex
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        tex_path.name
    ]

    subprocess.run(
        cmd,
        cwd=output_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return str(output_pdf_path)

