#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/core/utils.py
-------------------------------------------------------------------------------
Módulo de utilidades generales para el sistema Neural Networks
-------------------------------------------------------------------------------

Descripción general
-------------------
Este módulo contiene funciones auxiliares utilizadas en diferentes fases del
sistema: carga de datasets, preprocesamiento, normalización de símbolos
lógicos, análisis de configuración e impresión de estructuras internas.

Su propósito es garantizar que:
  • Los encabezados de columnas provenientes de archivos .ods/.xlsx/.csv
    se traduzcan a nombres ASCII consistentes.
  • Los símbolos lógicos (↔, ⊕, ∧, ∨, etc.) puedan ser interpretados de
    forma uniforme por el perceptrón.
  • Los parámetros booleanos del input.txt se interpreten de forma robusta.
  • La impresión de diccionarios internos sea compacta y amigable.

Este módulo es consumido por:
  - src/core/config.py                → parser de input.txt
  - src/core/data_extractor/*         → loader y preprocesamiento numérico
  - src/nn/perceptron.py              → entrenamiento del perceptrón
  - src/report/*                      → generación dinámica de reportes LaTeX

-------------------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Dict, Any


# =============================================================================
# === MAPA DE SÍMBOLOS LÓGICOS NORMALIZADOS ==================================
# =============================================================================
"""
Este diccionario permite convertir múltiples representaciones tipográficas
y textuales de operadores lógicos en un identificador ASCII estándar.

Ejemplos:

    • "↔", "⇔", "<->", "<=>"       → "BICONDITIONAL"
    • "⊕", "^", "exclusive-or"     → "XOR"
    • "∧", "&", "&&"               → "AND"
    • "∨", "|"                     → "OR"
    • "¬", "~", "!"                → "NOT"
    • "→", "=>"                    → "IMPLIES"

Esto es necesario porque los datasets pueden contener encabezados expresados
con símbolos Unicode, pero internamente es preferible trabajar con nombres
ASCII uniformes.
"""
LOGIC_SYMBOL_MAP = {

    # -------------------------------------------------------------------------
    # BICONDITIONAL  (↔, ⇔, ≡, <=>, ...)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # XOR (⊕, ⊻, ^, exclusive-or, ...)
    # -------------------------------------------------------------------------
    "⊕": "XOR",
    "⨁": "XOR",
    "⊻": "XOR",
    "^": "XOR",

    "xor": "XOR",
    "exor": "XOR",
    "exclusive_or": "XOR",
    "exclusive-or": "XOR",
    "exclusive or": "XOR",

    # -------------------------------------------------------------------------
    # AND (∧, &, &&, ...)
    # -------------------------------------------------------------------------
    "∧": "AND",
    "&": "AND",
    "&&": "AND",
    "⋀": "AND",

    "and": "AND",
    "logical_and": "AND",
    "logical-and": "AND",
    "logical and": "AND",

    # -------------------------------------------------------------------------
    # OR (∨, |, ||, ...)
    # -------------------------------------------------------------------------
    "∨": "OR",
    "|": "OR",
    "||": "OR",
    "⋁": "OR",

    "or": "OR",
    "logical_or": "OR",
    "logical-or": "OR",
    "logical or": "OR",

    # -------------------------------------------------------------------------
    # NOT (¬, ~, !)
    # -------------------------------------------------------------------------
    "¬": "NOT",
    "~": "NOT",
    "!": "NOT",

    "not": "NOT",
    "neg": "NOT",
    "negation": "NOT",

    # -------------------------------------------------------------------------
    # IMPLICATION (→, ⇒, ->, =>, ...)
    # -------------------------------------------------------------------------
    "→": "IMPLIES",
    "⇒": "IMPLIES",
    "⟹": "IMPLIES",

    "->": "IMPLIES",
    "=>": "IMPLIES",
    "-->": "IMPLIES",

    "implies": "IMPLIES",
    "implication": "IMPLIES",

    # -------------------------------------------------------------------------
    # NAND
    # -------------------------------------------------------------------------
    "↑": "NAND",
    "nand": "NAND",
    "!and": "NAND",
    "not and": "NAND",

    # -------------------------------------------------------------------------
    # NOR
    # -------------------------------------------------------------------------
    "↓": "NOR",
    "nor": "NOR",
    "!or": "NOR",
    "not or": "NOR",
}


# =============================================================================
# === NORMALIZACIÓN DE SÍMBOLOS LÓGICOS ======================================
# =============================================================================

def normalize_logic_symbol(name: str) -> str:
    """
    Normaliza un símbolo lógico (Unicode o textual) a un identificador ASCII.

    Parámetros
    ----------
    name : str
        Cadena original proveniente del dataset o del input.txt.

    Retorna
    -------
    str
        Símbolo normalizado según LOGIC_SYMBOL_MAP, o el original si
        no está contemplado en el mapa.
    """
    if not isinstance(name, str):
        return name

    # Limpieza base
    cleaned = name.strip().lower()
    cleaned = " ".join(cleaned.split())

    # Coincidencia directa
    if cleaned in LOGIC_SYMBOL_MAP:
        return LOGIC_SYMBOL_MAP[cleaned]

    # Quitar espacios internos
    cleaned_no_space = cleaned.replace(" ", "")
    if cleaned_no_space in LOGIC_SYMBOL_MAP:
        return LOGIC_SYMBOL_MAP[cleaned_no_space]

    # Quitar símbolos '-' y '_' para permitir variantes como "exclusive_or"
    cleaned_simple = cleaned.replace("-", "").replace("_", "")
    if cleaned_simple in LOGIC_SYMBOL_MAP:
        return LOGIC_SYMBOL_MAP[cleaned_simple]

    return name


# =============================================================================
# === NORMALIZACIÓN DE NOMBRES DE COLUMNAS ===================================
# =============================================================================

def normalize_colnames(cols) -> list[str]:
    """
    Normaliza una lista de nombres de columnas provenientes de un DataFrame.

    - Si un nombre es un operador lógico Unicode o ASCII → lo convierte.
    - Si no corresponde a ningún símbolo lógico → lo mantiene igual.

    Parámetros
    ----------
    cols : iterable
        Colección de nombres de columnas.

    Retorna
    -------
    list[str]
        Nombres normalizados.
    """
    normalized = []
    for c in cols:
        normalized.append(normalize_logic_symbol(str(c)))
    return normalized


# =============================================================================
# === CONVERSIÓN ROBUSTA DE BOOLEANOS ========================================
# =============================================================================

def parse_bool(s: str) -> bool:
    """
    Convierte una cadena textual en un valor booleano estándar.

    True si s es una de:
        {"1", "true", "yes", "y", "t", "si", "sí"}

    False en cualquier otro caso.

    Esta función es útil para parámetros opcionales en el input.txt.
    """
    return str(s).strip().lower() in {"1", "true", "yes", "y", "t", "si", "sí"}


# =============================================================================
# === IMPRESIÓN COMPACTA DE DICCIONARIOS =====================================
# =============================================================================

def dotted(d: Dict[str, Any]) -> str:
    """
    Devuelve un diccionario en formato compacto `k=v, k=v, ...`.

    Útil para depuración y visualización resumida de bloques
    parseados en el input.txt.

    Parámetros
    ----------
    d : Dict[str, Any]
        Diccionario a imprimir.

    Retorna
    -------
    str
        Representación compacta.
    """
    return ", ".join(f"{k}={v}" for k, v in d.items())

