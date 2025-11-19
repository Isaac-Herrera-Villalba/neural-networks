#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/report/report_nn_builder.py
------------------------------------------------------------
Constructor central del reporte para:

 - Perceptrón
 - Regla Delta (ADALINE)
 - Backpropagation (MLP)

Recibe los bloques procesados por config.py (lista de dicts),
ejecuta cada red neuronal, produce el bloque LaTeX correspondiente
y finalmente genera un único PDF.
------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict

# NN modules
from src.nn.perceptron import train_perceptron
from src.nn.delta_rule import train_delta_rule
from src.nn.mlp_backprop import train_backpropagation

# Preprocesamiento
from src.core.data_extractor.loader import load_dataset
from src.core.data_extractor.preprocess_numeric import preprocess_numeric_dataset

# Latex blocks
from src.report.latex_perceptron import build_perceptron_block
from src.report.latex_delta import build_delta_rule_block
from src.report.latex_backprop import build_backprop_block

# PDF renderer
from src.report.report_latex import render_all_instances_pdf, escape_latex


# ============================================================
# Normalización de símbolos lógicos
# ============================================================

LOGIC_MAP = {
    "↔": "BICONDITIONAL",
    "<->": "BICONDITIONAL",
    "<=>": "BICONDITIONAL",
    "≡": "BICONDITIONAL",
    "⇔": "BICONDITIONAL",

    "⊕": "XOR",
    "xor": "XOR",
    "XOR": "XOR",

    "∧": "AND",
    "AND": "AND",

    "∨": "OR",
    "OR": "OR",
}


def normalize_symbol(s: str) -> str:
    s = s.strip()
    return LOGIC_MAP.get(s, s)


# ============================================================
# PROCESAR UN BLOQUE INDIVIDUAL
# ============================================================

def process_block(block: Dict) -> str:
    """
    Procesa un bloque del input.txt y devuelve el bloque LaTeX.
    Si ocurre un error → se lanza excepción para ser capturada arriba.
    """

    # ------------------------------------------------------------
    # Validación básica
    # ------------------------------------------------------------
    required = ["NN", "DATASET", "SHEET", "X_COLS", "Y_COL"]
    for r in required:
        if r not in block:
            raise RuntimeError(f"Falta parámetro obligatorio: {r}")

    method = block["NN"].strip().upper()
    dataset_path = block["DATASET"].strip()
    sheet = block["SHEET"].strip()

    # columnas X
    X_cols = [c.strip() for c in block["X_COLS"].split(",")]

    # columna Y
    Y_col = normalize_symbol(block["Y_COL"])

    # hiperparámetros comunes
    lr = float(block.get("LEARNING_RATE", 0.1))
    max_epochs = int(block.get("MAX_EPOCHS", 20))
    threshold = float(block.get("THRESHOLD", 0.0))
    hidden = int(block.get("HIDDEN_NEURONS", 2))

    # pesos iniciales
    init_w = None
    if "INITIAL_WEIGHTS" in block:
        init_w = np.array(
            [float(v) for v in block["INITIAL_WEIGHTS"].split(",")]
        )

    # ------------------------------------------------------------
    # Cargar dataset
    # ------------------------------------------------------------
    df = load_dataset(dataset_path, sheet=sheet)

    # ------------------------------------------------------------
    # Preprocesamiento (X, y)
    # ------------------------------------------------------------
    X, y = preprocess_numeric_dataset(df, X_cols, Y_col)

    # ------------------------------------------------------------
    # Selección del método NN
    # ------------------------------------------------------------
    if method == "PERCEPTRON":
        result = train_perceptron(
            X=X,
            y=y,
            learning_rate=lr,
            max_epochs=max_epochs,
            threshold=threshold,
            initial_weights=init_w
        )

        return build_perceptron_block(
            result=result,
            df=df,
            X_cols=X_cols,
            Y_col=Y_col,
            lr=lr,
            threshold=threshold,
            initial_weights=init_w,
            max_epochs=max_epochs,
        )

    elif method in ["DELTA_RULE", "DELTA", "ADALINE"]:
        result = train_delta_rule(
            X=X,
            y=y,
            learning_rate=lr,
            max_epochs=max_epochs,
            initial_weights=init_w
        )

        return build_delta_rule_block(
            result=result,
            df=df,
            X_cols=X_cols,
            Y_col=Y_col,
            lr=lr,
            initial_weights=init_w,
            max_epochs=max_epochs,
        )

    elif method in ["MLP", "BACKPROP", "BACKPROPAGATION"]:
        result = train_backpropagation(
            X=X,
            y=y,
            hidden_neurons=hidden,
            learning_rate=lr,
            max_epochs=max_epochs
        )

        return build_backprop_block(
            result=result,
            df=df,
            X_cols=X_cols,
            Y_col=Y_col,
            lr=lr,
            hidden_neurons=hidden,
            max_epochs=max_epochs
        )

    else:
        raise RuntimeError(f"Método NN desconocido: {method}")


# ============================================================
# ENSAMBLADO COMPLETO DEL REPORTE
# ============================================================

def build_full_report(blocks: List[Dict]):
    """
    Procesa secuencialmente todos los bloques y genera
    un único PDF final.
    """

    final_latex = ""
    pdf_path = "output/reporte_nn.pdf"

    print(f"[INFO] {len(blocks)} bloques detectados")

    # recorrer bloques
    for idx, blk in enumerate(blocks, start=1):
        print(f"\n=== BLOQUE {idx}: MÉTODO {blk.get('NN', '').upper()} ===")

        try:
            block_tex = process_block(blk)
            final_latex += block_tex + "\n\n"

        except Exception as e:
            print(f"[ERROR] Fallo generando LaTeX en bloque {idx}: {e}")

    if not final_latex.strip():
        print("[WARN] No hay contenido válido para PDF.")
        return

    print(f"[INFO] PDF final: {pdf_path}")
    render_all_instances_pdf(pdf_path, final_latex)

    return final_latex, pdf_path

