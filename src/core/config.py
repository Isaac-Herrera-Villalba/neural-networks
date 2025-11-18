#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/core/config.py
------------------------------------------------------------
Descripción:
Parser de configuración para el proyecto de Redes Neuronales.

Este módulo lee el archivo `input.txt` y produce una lista de
bloques independientes, cada uno con su propio conjunto de
parámetros:

    DATASET       -> ruta al archivo de datos (.csv, .ods, .xlsx)
    SHEET         -> nombre de la hoja de cálculo
    METHOD        -> PERCEPTRON | DELTA_RULE | MLP_BACKPROP
    X_COLS        -> lista de columnas de entrada (X1, X2, ...)
    Y_COL         -> columna objetivo
    LEARNING_RATE -> tasa de aprendizaje
    MAX_EPOCHS    -> número máximo de iteraciones
    ERROR_THRESHOLD -> usado sólo en DELTA_RULE o MLP_BACKPROP

El parser conserva:
- comentarios de línea (#)
- comentarios de bloque (/* ... */)
- orden de los bloques

Además, normaliza automáticamente los operadores lógicos
presentes en X_COLS y Y_COL mediante:

    normalize_logic_symbol()

De esta forma, columnas como "↔", "⊕", "∨", "<=>" se transforman
internamente en:

    BICONDITIONAL, XOR, OR, ...

pero sin modificar el archivo .ods original.

Este módulo es consumido por:
- src/main.py
- src/nn/*   (perceptrón, delta, backprop)
- src/report/report_nn_builder.py
------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

from src.core.utils import normalize_logic_symbol, parse_bool


# ============================================================
# === CLASE PRINCIPAL ========================================
# ============================================================

class Config:
    """
    Gestiona la lectura de un archivo `input.txt` que puede
    contener múltiples bloques de configuración independientes
    para diferentes métodos de redes neuronales.
    """

    def __init__(self, path: str):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de entrada: {path}")

        # Lista de bloques en orden
        self.blocks: List[Dict] = []

        self._parse()

    # ========================================================
    # === PARSEO DE BLOQUES ================================
    # ========================================================
    def _parse(self):
        """
        Lee línea por línea, soporta:
        - comentarios de línea "#"
        - comentarios de bloque "/* ... */"
        - múltiples bloques independientes
        """
        with self.path.open("r", encoding="utf-8") as f:
            raw_lines = [l.rstrip("\n") for l in f.readlines()]

        lines: List[str] = []
        in_block_comment = False

        # ---- Fase 1: eliminar comentarios sin perder estructura ----
        for raw in raw_lines:
            line = raw.strip()

            # bloque /* ... */
            if "/*" in line and "*/" in line:
                # comentario en una sola línea
                before = line.split("/*", 1)[0]
                after = line.split("*/", 1)[1]
                line = (before + after).strip()

            elif "/*" in line:
                in_block_comment = True
                line = line.split("/*", 1)[0].strip()

            elif "*/" in line:
                in_block_comment = False
                line = line.split("*/", 1)[1].strip()

            if in_block_comment:
                continue

            # comentario de línea
            if line.startswith("#"):
                continue

            if line != "":
                lines.append(line)

        # ---- Fase 2: construcción de bloques ----
        current_block: Optional[Dict] = None

        for line in lines:

            # Inicio de un bloque: DATASET =
            if line.upper().startswith("DATASET"):
                # cerrar el bloque previo
                if current_block:
                    self._finalize_block(current_block)
                    self.blocks.append(current_block)

                current_block = {"KV": {}}
                key, val = line.split("=", 1)
                current_block["KV"][key.strip().upper()] = val.strip()
                continue

            if "=" in line:
                key, val = [x.strip() for x in line.split("=", 1)]
                if current_block is None:
                    continue
                current_block["KV"][key.upper()] = val
                continue

        # último bloque
        if current_block:
            self._finalize_block(current_block)
            self.blocks.append(current_block)

    # ========================================================
    # === NORMALIZACIÓN DE UN BLOQUE ========================
    # ========================================================
    def _finalize_block(self, block: Dict):
        """
        Aplica normalización adicional:
        - Uppercase de claves
        - Normalización de X_COLS y Y_COL
        - Validación básica
        """
        kv = block["KV"]

        # ---- Normalización de columnas X ----
        if "X_COLS" in kv:
            cols_raw = [c.strip() for c in kv["X_COLS"].split(",")]
            block["X_COLS"] = [
                normalize_logic_symbol(col) for col in cols_raw
            ]
        else:
            block["X_COLS"] = []

        # ---- Normalización de columna Y ----
        if "Y_COL" in kv:
            block["Y_COL"] = normalize_logic_symbol(kv["Y_COL"].strip())
        else:
            block["Y_COL"] = None

        # ---- Normalización de METHOD ----
        if "METHOD" in kv:
            method = kv["METHOD"].strip().upper()
            if method not in {"PERCEPTRON", "DELTA_RULE", "MLP_BACKPROP"}:
                raise ValueError(
                    f"[ERROR] Método desconocido: {method}. "
                    f"Use PERCEPTRON, DELTA_RULE o MLP_BACKPROP."
                )
            block["METHOD"] = method
        else:
            raise ValueError("Falta el parámetro METHOD en un bloque.")

        # ---- Parámetros numéricos opcionales ----
        def get_float(name, default=None):
            val = kv.get(name.upper(), None)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                raise ValueError(f"Parámetro inválido {name}: {val}")

        block["LEARNING_RATE"] = get_float("LEARNING_RATE", 0.5)
        block["MAX_EPOCHS"] = int(get_float("MAX_EPOCHS", 50))
        block["ERROR_THRESHOLD"] = get_float("ERROR_THRESHOLD", 0.001)

        # ---- Otros parámetros ----
        block["DATASET"] = kv.get("DATASET")
        block["SHEET"] = kv.get("SHEET")

    # ========================================================
    # === ACCESOR PÚBLICO ====================================
    # ========================================================
    def get_blocks(self) -> List[Dict]:
        """Devuelve la lista completa de bloques ya normalizados."""
        return self.blocks

