#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/report/report_nn_builder.py
-------------------------------------------------------------------------------
Constructor central del reporte final del proyecto.

Este módulo coordina todo el flujo de ejecución para cada bloque definido
en input.txt:

    INPUT  →  load_dataset()  →  preprocess_numeric_dataset()
           →  train_perceptron()  →  build_perceptron_block()
           →  ensamblado final LaTeX  →  render_all_instances_pdf()

-------------------------------------------------------------------------------
Ámbito del proyecto
-------------------
El curso oficialmente abarca únicamente el *Perceptrón simple* de Rosenblatt.

-------------------------------------------------------------------------------
Responsabilidades de este módulo
--------------------------------
1. Interpretar cada bloque del archivo de configuración input.txt.
2. Validar parámetros requeridos (NN, DATASET, SHEET, X_COLS, Y_COL, etc.).
3. Cargar el dataset correspondiente usando loader.py.
4. Preprocesar numéricamente las columnas indicadas.
5. Entrenar el perceptrón simple con los hiperparámetros dados.
6. Generar el bloque LaTeX correspondiente vía latex_perceptron.py.
7. Unir todos los bloques generados en un solo documento PDF.

Este módulo NO realiza cálculo matemático ni manipulación directa del modelo.
-------------------------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict

# NN: solo perceptrón
from src.nn.perceptron import train_perceptron

# Preprocesamiento
from src.core.data_extractor.loader import load_dataset
from src.core.data_extractor.preprocess_numeric import preprocess_numeric_dataset

# Bloques LaTeX permitidos
from src.report.latex_perceptron import build_perceptron_block

# Render PDF final
from src.report.report_latex import render_all_instances_pdf



# =============================================================================
# FUNCIÓN: process_block
# =============================================================================

def process_block(block: Dict) -> str:
    """
    Procesa un bloque del input.txt y retorna el LaTeX correspondiente.

    Flujo:
        1. Validar claves necesarias del bloque.
        2. Verificar que el método sea PERCEPTRON.
        3. Cargar dataset desde .ods/.xlsx/.csv.
        4. Preprocesar columnas X y Y:
              - conversión numérica,
              - mapeos a {-1, 1},
              - resolución robusta de nombres.
        5. Entrenar perceptrón simple.
        6. Construir bloque LaTeX.

    Parámetros
    ----------
    block : Dict
        Diccionario con las claves del bloque (ya limpiado por Config).

    Retorna
    -------
    str
        Código LaTeX generado para este bloque.
    """

    # --- Validación de claves obligatorias ---
    required = ["NN", "DATASET", "SHEET", "X_COLS", "Y_COL"]
    for r in required:
        if r not in block:
            raise RuntimeError(f"Falta parámetro obligatorio: {r}")

    method = block["NN"].strip().upper()
    if method != "PERCEPTRON":
        raise RuntimeError(
            f"Método '{method}' no soportado. "
            "Este proyecto solo implementa Perceptrón simple."
        )

    # --- Lectura de parámetros ---
    dataset_path = block["DATASET"].strip()
    sheet = block["SHEET"].strip()

    X_cols = [c.strip() for c in block["X_COLS"].split(",")]
    Y_col = block["Y_COL"].strip()

    lr = float(block.get("LEARNING_RATE", 0.1))
    max_epochs = int(block.get("MAX_EPOCHS", 20))
    threshold = float(block.get("THRESHOLD", 0.0))

    # Pesos iniciales opcionales
    init_w = None
    if "INITIAL_WEIGHTS" in block:
        init_w = np.array([
            float(v)
            for v in block["INITIAL_WEIGHTS"].split(",")
        ])

    # --- Cargar dataset ---
    df = load_dataset(dataset_path, sheet=sheet)

    # --- Preprocesar columnas ---
    X, y = preprocess_numeric_dataset(df, X_cols, Y_col)

    # --- Entrenar perceptrón simple ---
    result = train_perceptron(
        X=X,
        y=y,
        learning_rate=lr,
        max_epochs=max_epochs,
        threshold=threshold,
        initial_weights=init_w
    )

    # --- Construir bloque LaTeX del perceptrón ---
    return build_perceptron_block(
        result=result,
        df=df,
        X_cols=X_cols,
        Y_col=Y_col,
        lr=lr,
        threshold=threshold,
        initial_weights=init_w,
        max_epochs=max_epochs
    )



# =============================================================================
# FUNCIÓN: build_full_report
# =============================================================================

def build_full_report(blocks: List[Dict]):
    """
    Procesa todos los bloques PERCEPTRON y genera un PDF consolidado.

    Parámetros
    ----------
    blocks : List[Dict]
        Lista de diccionarios producidos por Config.get_blocks(),
        donde cada diccionario representa un bloque de input.txt.

    Retorna
    -------
    (latex, pdf_path) : Tuple[str, str]
        - Código LaTeX generado.
        - Ruta final del PDF.
    """

    final_latex = ""
    pdf_path = "output/reporte_nn.pdf"

    print(f"[INFO] {len(blocks)} bloques detectados")

    # Procesar secuencialmente cada bloque
    for idx, blk in enumerate(blocks, start=1):
        print(f"\n=== BLOQUE {idx}: MÉTODO {blk.get('NN', '').upper()} ===")

        try:
            block_tex = process_block(blk)
            final_latex += block_tex + "\n\n"
        except Exception as e:
            print(f"[ERROR] Fallo generando bloque {idx}: {e}")

    if not final_latex.strip():
        print("[WARN] No se produjo contenido LaTeX.")
        return

    print(f"[INFO] PDF final: {pdf_path}")
    render_all_instances_pdf(pdf_path, final_latex)

    return final_latex, pdf_path

