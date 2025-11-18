#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/core/utils.py
------------------------------------------------------------
Descripción:
Módulo de utilidades generales para el sistema de Redes
Neuronales y módulos auxiliares relacionados.

Incluye:
- Normalización de nombres de columnas.
- Normalización de símbolos lógicos (Unicode, ASCII, variantes
  tipográficas y textuales).
- Conversión robusta de valores booleanos.
- Impresión compacta de diccionarios.

Este módulo es consumido por:
- src/core/config.py           (parser de input.txt)
- src/core/data_extractor      (loader y preprocesamiento)
- src/nn/                      (módulos perceptrón / delta / MLP)
- src/report/                  (generación LaTeX)

Este módulo garantiza que los símbolos lógicos presentes en:
- columnas de .ods / .xlsx
- parámetros X_COLS y Y_COLS en input.txt

se conviertan a identificadores ASCII seguros sin modificar los
archivos originales.
------------------------------------------------------------
"""

from __future__ import annotations
from typing import Dict, Any


# ============================================================
# === MAPA DE SÍMBOLOS LÓGICOS NORMALIZADOS ==================
# ============================================================

LOGIC_SYMBOL_MAP = {

    # ===============================================================
    # BICONDITIONAL  (↔, <->, <=>, ⇔, ≡, ⟺, ⟷)
    # ===============================================================
    "↔": "BICONDITIONAL",
    "⇔": "BICONDITIONAL",
    "⇿": "BICONDITIONAL",
    "⟺": "BICONDITIONAL",
    "⟷": "BICONDITIONAL",
    "≡": "BICONDITIONAL",

    "<->": "BICONDITIONAL",
    "<=>": "BICONDITIONAL",
    "<==>": "BICONDITIONAL",
    "<-->": "BICONDITIONAL",

    "iff": "BICONDITIONAL",
    "equiv": "BICONDITIONAL",
    "biimplication": "BICONDITIONAL",
    "bi-implication": "BICONDITIONAL",
    "bi_implication": "BICONDITIONAL",
    "bimp": "BICONDITIONAL",
    "eqv": "BICONDITIONAL",
    "xnor": "BICONDITIONAL",

    # ===============================================================
    # XOR (⊕, ⊻, ^, exclusive-or)
    # ===============================================================
    "⊕": "XOR",
    "⨁": "XOR",
    "⊻": "XOR",
    "^": "XOR",

    "xor": "XOR",
    "exor": "XOR",
    "exclusive_or": "XOR",
    "exclusive-or": "XOR",
    "exclusive or": "XOR",

    # ===============================================================
    # AND (∧, &, &&, ⋀)
    # ===============================================================
    "∧": "AND",
    "&": "AND",
    "&&": "AND",
    "⋀": "AND",

    "and": "AND",
    "logical_and": "AND",
    "logical-and": "AND",
    "logical and": "AND",

    # ===============================================================
    # OR (∨, |, ||, ⋁)
    # ===============================================================
    "∨": "OR",
    "|": "OR",
    "||": "OR",
    "⋁": "OR",

    "or": "OR",
    "logical_or": "OR",
    "logical-or": "OR",
    "logical or": "OR",

    # ===============================================================
    # NOT (¬, ~, !, neg)
    # ===============================================================
    "¬": "NOT",
    "~": "NOT",
    "!": "NOT",

    "not": "NOT",
    "neg": "NOT",
    "negation": "NOT",

    # ===============================================================
    # IMPLICATION  (→, ⇒, ⟹, ->, =>, -->)
    # ===============================================================
    "→": "IMPLIES",
    "⇒": "IMPLIES",
    "⟹": "IMPLIES",

    "->": "IMPLIES",
    "=>": "IMPLIES",
    "-->": "IMPLIES",

    "implies": "IMPLIES",
    "implication": "IMPLIES",

    # ===============================================================
    # NAND
    # ===============================================================
    "↑": "NAND",
    "nand": "NAND",
    "!and": "NAND",
    "not and": "NAND",

    # ===============================================================
    # NOR
    # ===============================================================
    "↓": "NOR",
    "nor": "NOR",
    "!or": "NOR",
    "not or": "NOR",
}


# ============================================================
# === FUNCIÓN: Normalización de símbolos lógicos ==============
# ============================================================

def normalize_logic_symbol(name: str) -> str:
    """
    Normaliza símbolos lógicos o etiquetas provenientes del dataset o input.txt,
    convirtiéndolos a nombres ASCII seguros.

    Ejemplos:
        ↔     -> BICONDITIONAL
        <=>   -> BICONDITIONAL
        ⊕     -> XOR
        ∧     -> AND
        ¬     -> NOT
        →     -> IMPLIES

    Parámetros
    ----------
    name : str
        Nombre original de la columna o parámetro.

    Retorna
    -------
    str
        Nombre ASCII seguro normalizado, o el original si no pertenece al mapa.
    """
    if not isinstance(name, str):
        return name

    cleaned = name.strip().lower()
    cleaned = " ".join(cleaned.split())

    # coincidencia exacta
    if cleaned in LOGIC_SYMBOL_MAP:
        return LOGIC_SYMBOL_MAP[cleaned]

    # eliminar espacios internos
    cleaned_no_space = cleaned.replace(" ", "")
    if cleaned_no_space in LOGIC_SYMBOL_MAP:
        return LOGIC_SYMBOL_MAP[cleaned_no_space]

    # eliminar '-' y '_'
    cleaned_simple = cleaned.replace("-", "").replace("_", "")
    if cleaned_simple in LOGIC_SYMBOL_MAP:
        return LOGIC_SYMBOL_MAP[cleaned_simple]

    return name


# ============================================================
# === FUNCIÓN: Normalización de nombres de columnas ===========
# ============================================================

def normalize_colnames(cols) -> list[str]:
    """
    Normaliza nombres de columnas del dataset.

    - Aplica normalización lógica si corresponde.
    - Mantiene el nombre original si no es un símbolo lógico.
    - Evita modificar el archivo .ods, solo actúa en memoria.

    Parámetros
    ----------
    cols : iterable
        Lista de columnas del DataFrame.

    Retorna
    -------
    list[str]
        Lista procesada de nombres.
    """
    normalized = []
    for c in cols:
        c2 = normalize_logic_symbol(str(c))
        normalized.append(c2)
    return normalized


# ============================================================
# === FUNCIÓN: Conversión robusta de booleanos ===============
# ============================================================

def parse_bool(s: str) -> bool:
    """
    Convierte cadena textual a booleano.

    True si:
        "1", "true", "yes", "y", "t", "si", "sí"

    False en cualquier otro caso.
    """
    return str(s).strip().lower() in {"1", "true", "yes", "y", "t", "si", "sí"}


# ============================================================
# === FUNCIÓN: Impresión compacta de diccionarios ============
# ============================================================

def dotted(d: Dict[str, Any]) -> str:
    """
    Imprime un diccionario como: k=v, k=v, ...

    Parámetros
    ----------
    d : Dict[str, Any]
        Diccionario a procesar.

    Retorna
    -------
    str
        Formato compacto de pares clave-valor.
    """
    return ", ".join(f"{k}={v}" for k, v in d.items())

