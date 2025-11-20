#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/main.py — Versión corregida
"""

from __future__ import annotations
import sys
from pathlib import Path

import locale
locale.setlocale(locale.LC_NUMERIC, 'C')   # <<< AQUI

from src.core.config import Config
from src.report.report_nn_builder import build_full_report
from src.report.report_latex import render_all_instances_pdf


def main():

    cfg_path = Path("input.txt")
    if not cfg_path.exists():
        print("[ERROR] No existe input.txt")
        sys.exit(1)

    # Leer input.txt
    config = Config(str(cfg_path))
    blocks = config.get_blocks()

    print("\n=== DEBUG: BLOQUES PARSEADOS ===")
    import pprint
    pprint.pp(blocks)
    print("================================\n")


    if not blocks:
        print("[ERROR] No se detectaron bloques válidos.")
        sys.exit(1)

    print(f"[INFO] {len(blocks)} bloques detectados")

    # Construir reporte completo
    try:
        latex_code, pdf_path = build_full_report(blocks)
    except Exception as e:
        print(f"[ERROR] Fallo generando LaTeX: {e}")
        sys.exit(1)

    if not latex_code.strip():
        print("[WARN] No se produjo contenido LaTeX.")
        sys.exit(1)

    print(f"[INFO] Generando PDF final: {pdf_path}")

    try:
        render_all_instances_pdf(pdf_path, latex_code)
    except Exception as e:
        print(f"[ERROR] No se pudo compilar el PDF: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)

