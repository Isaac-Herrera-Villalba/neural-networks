#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/config.py
------------------------------------------------------------
Parser definitivo del input.txt para Neural Networks.

✔ Maneja comentarios (#, //, /* */)
✔ Soporta múltiples bloques
✔ Limpia inline comments
✔ Expande claves automáticamente (ya no usa "KV")
✔ Compatible con main.py
------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List


class Config:

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {path}")

        self.blocks: List[Dict] = []
        self._parse()

    # ------------------------------------------------------------
    def _strip_inline_comment(self, line: str) -> str:
        """Remueve comentarios después del valor."""
        if "#" in line:
            line = line.split("#", 1)[0]
        if "//" in line:
            line = line.split("//", 1)[0]
        return line.strip()

    # ------------------------------------------------------------
    def _parse(self):

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

            # Línea comentario normal
            if line.startswith("#"):
                continue

            # ------------------------------------------------------
            # Nuevo bloque: NN = ...
            # ------------------------------------------------------
            if line.upper().startswith("NN"):
                if current_block:
                    blocks.append(current_block)

                key, val = line.split("=", 1)
                current_block = {}
                current_block[key.strip()] = val.strip()
                continue

            # ------------------------------------------------------
            # Entrada clave = valor
            # ------------------------------------------------------
            if "=" in line:
                if current_block is None:
                    raise ValueError(
                        f"Error: se encontró '{line}' antes de definir un bloque 'NN = ...'."
                    )

                key, val = line.split("=", 1)
                key = key.strip()
                val = self._strip_inline_comment(val).strip()
                current_block[key] = val
                continue

        # Último bloque
        if current_block:
            blocks.append(current_block)

        self.blocks = blocks

    # ------------------------------------------------------------
    def get_blocks(self) -> List[Dict]:
        return self.blocks

