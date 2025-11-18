#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/main.py
------------------------------------------------------------
Punto de entrada principal del proyecto:

    ✦ Redes Neuronales (Perceptrón, Regla Delta, Backpropagation)

Flujo general:
------------------------------------------------------------
1. Leer input.txt utilizando Config.
2. Para cada bloque:
       - cargar dataset (CSV/XLSX/ODS)
       - preprocesar columnas X e Y
       - ejecutar el método solicitado (PERCEPTRON / DELTA / BACKPROP)
       - generar sección LaTeX específica
3. Unir todas las secciones
4. Generar el PDF final con report_latex.render_pdf()

Este archivo NO realiza cálculos matemáticos.
Toda la lógica está delegada a:

    core.config
    core.data_extractor.loader
    core.data_extractor.preprocess_numeric
    nn.perceptron
    nn.delta_rule
    nn.mlp_backprop
    report.latex_*
    report.report_nn_builder

------------------------------------------------------------
"""

from __future__ import annotations
import sys
from pathlib import Path

from src.core.config import Config
from src.core.data_extractor.loader import load_dataset
from src.core.data_extractor.preprocess_numeric import ensure_numeric_subset
from src.report.report_nn_builder import build_full_report


def main():
    """
    Ejecuta todo el flujo del proyecto de Redes Neuronales.
    Lee input.txt, procesa los bloques y genera el PDF final.
    """

    cfg_path = Path("input.txt")

    if not cfg_path.exists():
        print(f"[ERROR] No se encontró el archivo de configuración: {cfg_path}")
        sys.exit(1)

    # ========================================================
    # 1. Cargar input.txt
    # ========================================================
    try:
        config = Config(str(cfg_path))
    except Exception as e:
        print(f"[ERROR] No se pudo procesar input.txt: {e}")
        sys.exit(1)

    blocks = config.get_blocks()

    if not blocks:
        print("[ERROR] No se detectaron bloques en input.txt.")
        sys.exit(1)

    print(f"[INFO] {len(blocks)} bloques detectados en input.txt")

    # ========================================================
    # 2. Determinar ruta final del PDF
    #    (se respeta el último bloque que defina REPORT=)
    # ========================================================
    final_pdf = "output/reporte_nn.pdf"

    for blk in blocks:
        kv = blk["KV"]
        if "REPORT" in kv:
            final_pdf = kv["REPORT"]

    print(f"[INFO] PDF final: {final_pdf}")

    # ========================================================
    # 3. Generar reporte completo
    # ========================================================
    try:
        build_full_report(
            blocks=blocks,
            output_pdf_path=final_pdf,
            loader_fn=load_dataset,
            preprocess_fn=ensure_numeric_subset,
        )
    except Exception as e:
        print(f"[ERROR] Fallo durante la generación del reporte: {e}")
        sys.exit(1)

    print("[OK] Proceso completado correctamente.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ejecución cancelada por el usuario.")
        sys.exit(130)

