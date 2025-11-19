#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/data_extractor/preprocess_numeric.py
------------------------------------------------------------
Módulo de preprocesamiento para Redes Neuronales.

Función principal:
    preprocess_numeric_dataset(df, x_cols, y_col)

Responsabilidades:
  - Resolver correctamente los nombres de columnas X e Y
    aunque difieran en mayúsculas/minúsculas respecto a input.txt.
  - Convertir las columnas X a valores numéricos (float).
  - Convertir la columna Y a valores en {-1, 1} a partir de:
        - 0 / 1 numéricos
        - "0" / "1" como texto
        - "true"/"false", "TRUE"/"FALSE"
  - Eliminar filas inválidas (NaN) en cualquiera de las columnas usadas.
  - Devolver X (np.ndarray) e y (np.ndarray columna).

NOTA:
  El mapeo de símbolos lógicos (↔, ⊕, ∧, ∨, etc.) se resuelve a nivel
  de configuración / builder (report_nn_builder.py) en el nombre de
  la columna Y. Aquí se asume que y_col ya es el nombre normalizado
  de la columna de salida en el DataFrame.
------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np


# ============================================================
#   MAPA DE VERDAD A {-1, 1}
# ============================================================

TRUTH_MAP = {
    "1": 1,
    "0": -1,
    "true": 1,
    "false": -1,
    "TRUE": 1,
    "FALSE": -1,
}


# ============================================================
#   RESOLVER NOMBRES DE COLUMNAS (CASE-INSENSITIVE)
# ============================================================

def _resolve_column_name(df: pd.DataFrame, logical_name: str) -> str:
    """
    Dado un nombre lógico (como aparece en input.txt),
    encuentra el nombre real de la columna en el DataFrame,
    ignorando mayúsculas/minúsculas y espacios alrededor.

    Ejemplo:
      logical_name = "AND"
      columnas df = ["x1", "x2", "and"]
      -> retorna "and"
    """
    target = str(logical_name).strip().casefold()
    for col in df.columns:
        if str(col).strip().casefold() == target:
            return col
    raise KeyError(f"Columna '{logical_name}' no encontrada en el DataFrame.")


# ============================================================
#   CONVERSIÓN DE Y A {-1, 1}
# ============================================================

def _convert_y_value(val):
    """
    Convierte un escalar de la columna Y a un valor numérico,
    preferentemente en el conjunto {-1, 1}.

    Reglas:
      - NaN -> None (para ser eliminado posteriormente)
      - "0" / 0  -> -1
      - "1" / 1  ->  1
      - "true"/"false" -> TRUTH_MAP
      - Otros valores numéricos -> float directo
      - Cualquier cosa no interpretable -> None
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None

    s = str(val).strip()

    # true/false explícitos
    if s in TRUTH_MAP:
        return float(TRUTH_MAP[s])

    # intento de número
    try:
        num = float(s)
    except ValueError:
        return None

    # normalizar 0/1 a -1/1
    if num == 0.0:
        return -1.0
    if num == 1.0:
        return 1.0

    # para otros valores numéricos, se retorna tal cual
    return num


# ============================================================
#   FUNCIÓN PRINCIPAL — PREPROCESAMIENTO
# ============================================================

def preprocess_numeric_dataset(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte las columnas indicadas en un dataset numérico
    apto para los algoritmos de redes neuronales.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame cargado (por ejemplo, desde .ods).
    x_cols : List[str]
        Nombres lógicos de las columnas de entrada X (como en input.txt).
    y_col : str
        Nombre lógico de la columna de salida Y (como en input.txt).

    Retorna
    -------
    X : np.ndarray
        Matriz de características, de forma (N, n_inputs).
    y : np.ndarray
        Vector columna con la salida, de forma (N, 1), con valores en {-1, 1}
        o, en general, numéricos válidos.
    """

    df2 = df.copy()

    # --------------------------------------------------------
    # Resolver nombres reales de columnas en el DataFrame
    # --------------------------------------------------------
    resolved_x_cols: List[str] = []
    for name in x_cols:
        resolved_x_cols.append(_resolve_column_name(df2, name))

    resolved_y_col = _resolve_column_name(df2, y_col)

    # --------------------------------------------------------
    # Convertir X a numérico (float)
    # --------------------------------------------------------
    try:
        for c in resolved_x_cols:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    except Exception as e:
        raise ValueError(f"Error convirtiendo columnas X a numérico: {e}")

    # --------------------------------------------------------
    # Convertir Y a valores numéricos (idealmente {-1, 1})
    # --------------------------------------------------------
    try:
        df2[resolved_y_col] = df2[resolved_y_col].apply(_convert_y_value)
    except Exception as e:
        raise ValueError(f"Error convirtiendo columna Y a numérico: {e}")

    # --------------------------------------------------------
    # Eliminar filas inválidas
    # --------------------------------------------------------
    before = len(df2)
    df2 = df2.dropna(subset=resolved_x_cols + [resolved_y_col])
    dropped = before - len(df2)

    if df2.empty:
        raise ValueError("No hay datos válidos después del preprocesamiento.")

    # --------------------------------------------------------
    # Construir X e y como arreglos NumPy
    # --------------------------------------------------------
    X = df2[resolved_x_cols].to_numpy(dtype=float)
    y = df2[resolved_y_col].to_numpy(dtype=float).reshape(-1, 1)

    return X, y

