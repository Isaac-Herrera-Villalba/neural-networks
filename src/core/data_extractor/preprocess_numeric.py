#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/data_extractor/preprocess_numeric.py
---------------------------------------------------------------------------
Módulo de preprocesamiento numérico para redes neuronales (Perceptrón Simple)
---------------------------------------------------------------------------

Objetivo general
----------------
Proveer una conversión robusta entre un DataFrame arbitrario (limpio,
producido por loader.py) y los tensores numéricos requeridos por los
algoritmos de entrenamiento (X e y).

Este módulo se asegura de que:

    1. Las columnas indicadas en input.txt (X_COLS y Y_COL)
       se resuelvan correctamente en el DataFrame, incluso si la
       capitalización difiere ("Y" ≠ "y" ≠ "y ").

    2. Todas las columnas X sean convertidas a valores numéricos float.
       Cualquier dato no interpretable → NaN → fila eliminada.

    3. La columna Y sea interpretada correctamente en sentido lógico:
           • 0 → -1
           • 1 →  1
           • "true"/"false" → 1 / -1
           • Valores numéricos arbitrarios → float directo

    4. Se eliminen filas inválidas (NaN) en X o Y.

    5. Se retornen:
           X → matriz NumPy de forma (N, n_features)
           y → vector columna (N, 1)

Este módulo **no normaliza símbolos lógicos como ↔, ⊕, ∧, ∨**.
Dicha normalización ocurre antes, durante el proceso de carga
(loader.py) o en report_nn_builder.py.

Se utiliza en:
    - src/report/report_nn_builder.py
    - src/nn/perceptron.py
    - src/main.py
---------------------------------------------------------------------------

"""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np


# ============================================================================
#   MAPA DE VALORES DE VERDAD A {-1, 1}
# ============================================================================
TRUTH_MAP = {
    "1": 1,
    "0": -1,
    "true": 1,
    "false": -1,
    "TRUE": 1,
    "FALSE": -1,
}


# ============================================================================
#   RESOLUCIÓN DE NOMBRES DE COLUMNAS (INSENSIBLE A MAYÚSC/MINÚSC)
# ============================================================================

def _resolve_column_name(df: pd.DataFrame, logical_name: str) -> str:
    """
    Resuelve un nombre lógico (proveniente de input.txt) a un nombre
    real dentro del DataFrame, ignorando mayúsculas/minúsculas.

    Ejemplo:
        input.txt → "AND"
        DataFrame tiene columnas ["x1", "x2", "and"]
        → retorna "and"

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame ya cargado y normalizado.
    logical_name : str
        Nombre escrito por el usuario en input.txt.

    Retorna
    -------
    str : nombre exacto de la columna dentro del DataFrame.

    Lanza
    -----
    KeyError si la columna no se encuentra.
    """
    target = str(logical_name).strip().casefold()
    for col in df.columns:
        if str(col).strip().casefold() == target:
            return col
    raise KeyError(f"Columna '{logical_name}' no encontrada en el DataFrame.")


# ============================================================================
#   CONVERSIÓN DE Y A VALORES NUMÉRICOS VÁLIDOS
# ============================================================================

def _convert_y_value(val):
    """
    Convierte un valor individual de la columna Y a un número válido,
    idealmente en el conjunto {-1, 1}.

    Reglas aplicadas:
        • NaN           → None  (fila eliminada posteriormente)
        • "0" / 0       → -1
        • "1" / 1       →  1
        • "true"/"false" (en distintos casos) → mapeo TRUTH_MAP
        • Números arbitrarios → se intentan convertir a float
        • Si no es interpretable → None

    Parámetros
    ----------
    val : any
        Valor crudo proveniente del DataFrame.

    Retorna
    -------
    float | None
        Valor numérico normalizado o None si inválido.
    """
    # Caso: valor nulo explícito
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None

    s = str(val).strip()

    # true/false explícitos
    if s in TRUTH_MAP:
        return float(TRUTH_MAP[s])

    # intento de conversión numérica
    try:
        num = float(s)
    except ValueError:
        return None

    # normalizar booleano 0/1 a -1/1
    if num == 0.0:
        return -1.0
    if num == 1.0:
        return 1.0

    # otros valores numéricos permitidos (aunque raros en lógica)
    return num


# ============================================================================
#   FUNCIÓN PRINCIPAL — PREPROCESAMIENTO COMPLETO
# ============================================================================

def preprocess_numeric_dataset(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesa un DataFrame para obtener matrices numéricas aptas para
    entrenamiento de redes neuronales.

    Esta es la función que debe usarse desde report_nn_builder.py.

    Flujo general:
    --------------
      1. Copiar el DataFrame para no modificar el original.
      2. Resolver nombres reales de columnas X e Y.
      3. Convertir columnas X a float.
      4. Convertir Y a {-1, 1} o números válidos.
      5. Eliminar filas con datos faltantes.
      6. Convertir a matrices NumPy.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame limpio, producido por loader.py.
    x_cols : List[str]
        Nombres lógicos de las columnas de entrada (como en input.txt).
    y_col : str
        Nombre lógico de la columna objetivo.

    Retorna
    -------
    X : np.ndarray de forma (N, n_inputs)
        Matriz numérica de características.
    y : np.ndarray de forma (N, 1)
        Vector columna numérico, típico en {-1, 1}.

    Lanza
    -----
    KeyError     : Si alguna columna no existe en el DataFrame.
    ValueError   : Si X o Y contienen valores no interpretable y luego
                   de eliminar filas no queda ningún dato válido.
    """

    df2 = df.copy()

    # --------------------------------------------------------
    # 1. Resolver los nombres de columnas X
    # --------------------------------------------------------
    resolved_x_cols: List[str] = []
    for name in x_cols:
        resolved_x_cols.append(_resolve_column_name(df2, name))

    # 2. Resolver la columna Y
    resolved_y_col = _resolve_column_name(df2, y_col)

    # --------------------------------------------------------
    # 3. Convertir columnas X a numérico (float)
    # --------------------------------------------------------
    try:
        for c in resolved_x_cols:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    except Exception as e:
        raise ValueError(f"Error convirtiendo columnas X a numérico: {e}")

    # --------------------------------------------------------
    # 4. Convertir Y a valores numéricos válidos
    # --------------------------------------------------------
    try:
        df2[resolved_y_col] = df2[resolved_y_col].apply(_convert_y_value)
    except Exception as e:
        raise ValueError(f"Error convirtiendo columna Y a numérico: {e}")

    # --------------------------------------------------------
    # 5. Eliminar filas inválidas (valores NaN o None)
    # --------------------------------------------------------
    before = len(df2)
    df2 = df2.dropna(subset=resolved_x_cols + [resolved_y_col])
    dropped = before - len(df2)

    if df2.empty:
        raise ValueError("No hay datos válidos después del preprocesamiento.")

    # --------------------------------------------------------
    # 6. Convertir a NumPy (X e y)
    # --------------------------------------------------------
    X = df2[resolved_x_cols].to_numpy(dtype=float)
    y = df2[resolved_y_col].to_numpy(dtype=float).reshape(-1, 1)

    return X, y

