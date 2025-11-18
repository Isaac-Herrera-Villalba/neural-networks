#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/core/data_extractor/preprocess_numeric.py
------------------------------------------------------------
Descripción:
Módulo encargado de convertir los valores del dataset a tipo
numérico (float) y eliminar cualquier fila con datos inválidos,
garantizando que los módulos de redes neuronales operen sobre
datos limpios y consistentes.

Este preprocesador es utilizado por:
    - Perceptrón (clasificación lineal)
    - Regla Delta (grado de error cuadrático)
    - MLP mediante Backpropagation

Funciones principales:
------------------------------------------------------------
1. Conversión segura de columnas numéricas:
       pd.to_numeric(..., errors="coerce")

2. Eliminación automática de filas con valores NaN.

3. Reporte del número de filas eliminadas.

4. Acepta cualquier cantidad de variables independientes X.

Notas:
------------------------------------------------------------
- Este módulo NO modifica encabezados (eso lo hace loader.py).
- NO interpreta operadores lógicos (ya normalizados por config &
  utils).
- Después de este paso, todos los valores del DataFrame son
  flotantes puros, listos para cálculos vectoriales y gradientes.
------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd


# ============================================================
# === FUNCIÓN PRINCIPAL ======================================
# ============================================================

def ensure_numeric_subset(
    df: pd.DataFrame,
    used_cols: List[str]
) -> Tuple[pd.DataFrame, int]:
    """
    Convierte a tipo float las columnas indicadas y elimina las filas
    con valores no convertibles.

    Parámetros
    ----------
    df : pd.DataFrame
        Conjunto de datos original (ya con columnas normalizadas).
    used_cols : List[str]
        Lista de columnas que deben convertirse a valores numéricos.
        Incluye típicamente: X_COLS + [Y_COL].

    Retorna
    -------
    Tuple[pd.DataFrame, int]
        df_clean : DataFrame depurado con valores numéricos reales.
        dropped  : número de filas eliminadas durante la limpieza.

    Flujo general
    -------------
    1. Copia del DataFrame para no modificar el original.
    2. Intento de conversión columna por columna.
    3. Reemplazo de valores inválidos por NaN.
    4. Eliminación de filas con NaN.
    5. Retorno del DataFrame limpio.
    """
    df2 = df.copy()

    # Intentar conversión a float columna por columna
    for col in used_cols:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    # Conteo antes/después para saber cuántas filas se eliminaron
    before = len(df2)
    df2 = df2.dropna(subset=used_cols).reset_index(drop=True)
    dropped = before - len(df2)

    return df2, dropped

