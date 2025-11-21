#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/main.py
-------------------------------------------------------------------------------
Punto de entrada principal del proyecto Neural Networks (Perceptrón Simple)
-------------------------------------------------------------------------------

Descripción general
-------------------
Este módulo implementa el flujo de ejecución completo del proyecto, actuando
como coordinador entre:

    • El parser del archivo input.txt
    • El cargador y preprocesador de datasets
    • El módulo de entrenamiento del perceptrón simple
    • El generador de bloques LaTeX por instancia
    • El ensamblador final del reporte académico en PDF

El archivo `input.txt` define múltiples bloques de configuración, cada uno de
los cuales representa una ejecución independiente del perceptrón sobre un
dataset y parámetros específicos.

Flujo del programa (pipeline general)
-------------------------------------
1. Verificar la existencia de input.txt.
2. Parsear el archivo mediante `Config`, obteniendo una lista de bloques.
3. Mostrar los bloques detectados para depuración.
4. Llamar a `build_full_report(blocks)`:
       - Procesa cada bloque.
       - Entrena el perceptrón.
       - Genera el bloque LaTeX correspondiente.
       - Ensambla todos los bloques en un único documento.
5. Ejecutar `pdflatex` para producir:
       output/reporte_nn.pdf
6. Manejo robusto de errores:
       - Faltas en el input.txt
       - Fallos en el entrenamiento
       - Errores al compilar LaTeX

Este módulo no contiene lógica matemática ni de redes neuronales. Su función
es **integrar** y **coordinar** los componentes del sistema.

Dependencias directas
---------------------
    - src/core/config.py
    - src/report/report_nn_builder.py
    - src/report/report_latex.py

Ejecutado como:
    python3 -m src.main
    o simplemente:
    make run
-------------------------------------------------------------------------------
"""

from __future__ import annotations
import sys
from pathlib import Path

import locale
locale.setlocale(locale.LC_NUMERIC, 'C')

from src.core.config import Config
from src.report.report_nn_builder import build_full_report
from src.report.report_latex import render_all_instances_pdf


def main():
    """
    Orquesta el pipeline completo del proyecto:

        1. Lee input.txt
        2. Aplica el parser de configuración
        3. Genera el LaTeX para cada bloque
        4. Ensambla el PDF final

    Maneja mensajes informativos, errores y salida adecuada para
    integración con el Makefile del proyecto.
    """

    cfg_path = Path("input.txt")
    if not cfg_path.exists():
        print("[ERROR] No existe input.txt")
        sys.exit(1)

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

    try:
        latex_code, pdf_path = build_full_report(blocks)
    except Exception as e:
        print(f"[ERROR] Fallo generando LaTeX: {e}")
        sys.exit(1)

    if not latex_code.strip():
        print("[WARN] No se generó contenido para el PDF.")
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

