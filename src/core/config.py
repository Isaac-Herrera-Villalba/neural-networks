#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/config.py
-------------------------------------------------------------------------------
Parser robusto para el archivo input.txt del proyecto Neural Networks
-------------------------------------------------------------------------------

Descripción general
-------------------
Este módulo implementa el analizador (parser) oficial del archivo
`input.txt`, utilizado exclusivamente para configurar **instancias de
Perceptrón Simple**, el único modelo incluido en la versión actual del
proyecto.

El parser interpreta cada bloque definido en input.txt, donde se especifica:

    • Qué dataset utilizar (DATASET = ...)
    • Qué hoja del archivo cargar (SHEET = ...)
    • Qué columnas corresponden a las entradas X (X_COLS = ...)
    • Qué columna corresponde a la salida Y (Y_COL = ...)
    • Parámetros del entrenamiento (LEARNING_RATE, MAX_EPOCHS, THRESHOLD, etc.)

Cada bloque representa una ejecución independiente del perceptrón simple.

Características principales
---------------------------
✔ Soporte para múltiples bloques consecutivos
✔ Soporta comentarios de línea:
        • #
        • //
✔ Soporta comentarios de bloque:
        • /* ... */
✔ Soporta comentarios en la misma línea de una asignación:
        • LEARNING_RATE = 0.5   # comentario
✔ Procesa claves de forma tolerante a espacios alrededor del '='
✔ Valida que cada bloque comience con `NN = PERCEPTRON`
✔ Produce una lista de diccionarios lista para ser consumida por:

        • report_nn_builder.py
        • main.py

Formato esperado del input.txt
------------------------------
Cada bloque debe comenzar de forma explícita con:

    NN = PERCEPTRON

y continuar con claves como:

    DATASET = data/dataset.ods
    SHEET   = Sheet1
    X_COLS  = x1, x2
    Y_COL   = AND

Los bloques adicionales comienzan simplemente con otro `NN = PERCEPTRON`.

Salida del parser
-----------------
`self.blocks` es una lista de diccionarios, por ejemplo:

    [
      {
        "NN": "PERCEPTRON",
        "DATASET": "data/dataset.ods",
        "SHEET": "Sheet1",
        "X_COLS": "x1, x2",
        "Y_COL": "AND",
        "LEARNING_RATE": "0.5",
        "MAX_EPOCHS": "30"
      },
      ...
    ]

-------------------------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List


class Config:
    """
    Lee y procesa el archivo input.txt generando una lista de bloques.
    Cada bloque describe una instancia independiente del perceptrón simple.
    """

    def __init__(self, path: str):
        """
        Constructor: valida la existencia del archivo y ejecuta el parseo.

        Parámetros
        ----------
        path : str
            Ruta al archivo input.txt.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {path}")

        self.blocks: List[Dict] = []
        self._parse()

    # ----------------------------------------------------------------------
    def _strip_inline_comment(self, line: str) -> str:
        """
        Elimina comentarios que aparecen en la misma línea.

        Ejemplos:
            LEARNING_RATE = 0.5   # comentario
            X_COLS = x1, x2       // comentario
        """
        if "#" in line:
            line = line.split("#", 1)[0]
        if "//" in line:
            line = line.split("//", 1)[0]
        return line.strip()

    # ----------------------------------------------------------------------
    def _parse(self):
        """
        Analiza todo el archivo input.txt y construye la lista de bloques.

        Reglas:
            • Ignora líneas vacías
            • Ignora comentarios de bloque /* ... */
            • Ignora comentarios de línea
            • Cada bloque inicia con "NN = PERCEPTRON"
            • Todas las demás claves se asocian al bloque actual
        """
        raw_lines = self.path.read_text(encoding="utf-8").splitlines()

        blocks = []
        current_block = None
        in_block_comment = False

        for raw in raw_lines:
            line = raw.strip()

            if not line:
                continue

            # Comentarios de bloque
            if "/*" in line and "*/" in line:
                continue
            if "/*" in line:
                in_block_comment = True
                continue
            if "*/" in line:
                in_block_comment = False
                continue
            if in_block_comment:
                continue

            # Comentarios simples
            if line.startswith("#"):
                continue

            # Inicio de bloque
            if line.upper().startswith("NN"):
                if current_block:
                    blocks.append(current_block)

                key, val = line.split("=", 1)
                current_block = {}
                current_block[key.strip()] = val.strip()
                continue

            # Asignación clave = valor
            if "=" in line:
                if current_block is None:
                    raise ValueError(
                        f"Error: se encontró '{line}' antes de definir un bloque 'NN = PERCEPTRON'."
                    )

                key, val = line.split("=", 1)
                key = key.strip()
                val = self._strip_inline_comment(val).strip()
                current_block[key] = val
                continue

        if current_block:
            blocks.append(current_block)

        self.blocks = blocks

    # ----------------------------------------------------------------------
    def get_blocks(self) -> List[Dict]:
        """
        Retorna la lista completa de bloques parseados.
        """
        return self.blocks

