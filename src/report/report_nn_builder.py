#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/report/report_nn_builder.py
------------------------------------------------------------
Descripción:
Constructor central del reporte LaTeX para el proyecto de Redes
Neuronales. Este módulo recibe:

  - Configuración ya parseada (bloques del input.txt)
  - Dataset numérico preprocesado
  - Resultado del entrenamiento (Perceptrón, Delta o Backprop)

Y genera un bloque LaTeX completo listo para ser enviado a:

    report_latex.render_pdf()

Este módulo NO realiza ningún cálculo de redes neuronales.
Solo coordina:

    core.config          → parámetros
    nn.perceptron        → entrenamiento
    nn.delta_rule        → entrenamiento
    nn.mlp_backprop      → entrenamiento
    report.latex_*       → generación LaTeX

------------------------------------------------------------
"""

from __future__ import annotations
from typing import Dict, List
import pandas as pd

# Entrenadores
from src.nn.perceptron import train_perceptron, PerceptronResult
from src.nn.delta_rule import train_delta_rule, DeltaRuleResult
from src.nn.mlp_backprop import train_backpropagation, MLPBackpropResult

# Bloques LaTeX específicos
from src.report.latex_perceptron import build_perceptron_block
from src.report.latex_delta import build_delta_rule_block
from src.report.latex_backprop import build_backprop_block

# Utilidades LaTeX
from src.report.report_latex import render_pdf


# ============================================================
# === CONSTRUCTOR PRINCIPAL DEL REPORTE ======================
# ============================================================

def build_full_report(
    blocks: List[Dict],
    output_pdf_path: str,
    loader_fn,
    preprocess_fn,
):
    """
    Construye el reporte completo en LaTeX para TODOS los bloques
    definidos en el input.txt. Cada bloque puede usar un método
    distinto: PERCEPTRON, DELTA o BACKPROP.

    Parámetros
    ----------
    blocks : list[dict]
        Bloques cargados desde config.Config.
    output_pdf_path : str
        Ruta del PDF final. El archivo .tex se generará junto al PDF.
    loader_fn : Callable
        Función para cargar dataset desde CSV/XLSX/ODS.
        (core.data_extractor.loader.load_dataset)
    preprocess_fn : Callable
        Función para limpiar columnas numéricas.
        (core.data_extractor.preprocess_numeric.ensure_numeric_subset)

    Flujo:
    ------
      Por cada BLOQUE:
        1. Cargar dataset
        2. Preprocesar X,Y
        3. Detectar método NN
        4. Ejecutar entrenamiento
        5. Generar bloque LaTeX del método
        6. Concatenar al documento

    Finalmente:
        render_pdf() genera el PDF real.
    """

    report_sections: List[str] = []

    # ========================================================
    # Procesar cada bloque del input.txt
    # ========================================================
    for idx, block in enumerate(blocks, start=1):
        kv = block["KV"]
        method = kv.get("METHOD", "").strip().upper()

        dataset_path = kv.get("DATASET")
        sheet_name = kv.get("SHEET")
        x_cols_raw = kv.get("X_COLS")
        y_col = kv.get("Y_COL")

        if not (dataset_path and x_cols_raw and y_col and method):
            print(f"[WARN] Bloque {idx}: configuración incompleta, se omite.")
            continue

        x_cols = [c.strip() for c in x_cols_raw.split(",")]

        print(f"\n=== BLOQUE {idx}: MÉTODO {method} ===")
        print(f"Dataset: {dataset_path}")
        print(f"Sheet:   {sheet_name}")
        print(f"Entradas X: {x_cols}")
        print(f"Salida Y:   {y_col}")

        # ====================================================
        # 1. CARGA DE DATASET
        # ====================================================
        try:
            df = loader_fn(dataset_path, sheet=sheet_name)
        except Exception as e:
            print(f"[ERROR] al cargar dataset: {e}")
            continue

        # ====================================================
        # 2. PREPROCESAMIENTO NUMÉRICO
        # ====================================================
        used_cols = x_cols + [y_col]
        df_num, dropped = preprocess_fn(df, used_cols)

        if df_num.empty:
            print(f"[WARN] Dataset vacío tras preprocesamiento.")
            continue

        if dropped > 0:
            print(f"[INFO] Filas eliminadas: {dropped}")

        # ====================================================
        # 3. DETECTAR MÉTODO
        # ====================================================
        section_title = kv.get("SECTION_TITLE", f"Bloque {idx}: {method}")

        if method == "PERCEPTRON":

            eta = float(kv.get("LEARNING_RATE", 0.5))
            max_epochs = int(kv.get("MAX_EPOCHS", 50))

            result: PerceptronResult = train_perceptron(
                df=df_num,
                x_cols=x_cols,
                y_col=y_col,
                learning_rate=eta,
                max_epochs=max_epochs,
            )

            latex_block = build_perceptron_block(
                df_num=df_num,
                x_cols=x_cols,
                y_col=y_col,
                learning_rate=eta,
                max_epochs=max_epochs,
                result=result,
                section_title=section_title,
            )

        elif method == "DELTA":

            eta = float(kv.get("LEARNING_RATE", 0.5))
            max_epochs = int(kv.get("MAX_EPOCHS", 100))
            eps = float(kv.get("ERROR_THRESHOLD", 0.01))

            result: DeltaRuleResult = train_delta_rule(
                df=df_num,
                x_cols=x_cols,
                y_col=y_col,
                learning_rate=eta,
                max_epochs=max_epochs,
                mse_threshold=eps,
            )

            latex_block = build_delta_rule_block(
                df_num=df_num,
                x_cols=x_cols,
                y_col=y_col,
                learning_rate=eta,
                max_epochs=max_epochs,
                error_threshold=eps,
                result=result,
                section_title=section_title,
            )

        elif method == "BACKPROP":

            eta = float(kv.get("LEARNING_RATE", 0.5))
            hidden = int(kv.get("HIDDEN_NEURONS", 2))
            max_epochs = int(kv.get("MAX_EPOCHS", 200))
            eps = float(kv.get("ERROR_THRESHOLD", 0.01))
            activation = kv.get("ACTIVATION", "SIGMOID")

            result: MLPBackpropResult = train_backpropagation(
                df=df_num,
                x_cols=x_cols,
                y_col=y_col,
                learning_rate=eta,
                hidden_neurons=hidden,
                max_epochs=max_epochs,
                mse_threshold=eps,
                activation=activation,
            )

            latex_block = build_backprop_block(
                df_num=df_num,
                x_cols=x_cols,
                y_col=y_col,
                learning_rate=eta,
                hidden_neurons=hidden,
                max_epochs=max_epochs,
                error_threshold=eps,
                activation_name=activation,
                result=result,
                section_title=section_title,
            )

        else:
            print(f"[ERROR] Método no reconocido en bloque {idx}: {method}")
            continue

        report_sections.append(latex_block)

    # ========================================================
    # 4. GENERAR PDF
    # ========================================================
    if report_sections:
        full_doc = "\n".join(report_sections)
        print("\n=== Generando PDF final ===")
        render_pdf(output_pdf_path, full_doc)
    else:
        print("[WARN] No se generó ningún bloque LaTeX.")

