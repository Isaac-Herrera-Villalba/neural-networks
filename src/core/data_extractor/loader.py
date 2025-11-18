#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/core/data_extractor/loader.py
------------------------------------------------------------
Descripción:
Módulo encargado de la carga, limpieza y normalización de
datasets en formatos:

    - .csv
    - .xlsx
    - .ods

El objetivo es producir un DataFrame limpio con encabezados
correctamente detectados y normalizados para uso interno en
los módulos de redes neuronales (perceptrón, regla delta,
MLP con backpropagation).

Características principales:
------------------------------------------------------------
1. Detección automática del bloque tabular real incluso si:
   - hay filas/columnas vacías al inicio,
   - el encabezado está desplazado,
   - existen celdas vacías.

2. Eliminación de filas y columnas totalmente vacías.

3. Normalización automática de encabezados mediante:

        normalize_colnames()
        normalize_logic_symbol()

   Lo que permite usar columnas como "↔", "⊕", "∨", "<=>", etc.,
   convirtiéndolas internamente en:

        BICONDITIONAL, XOR, OR, ...

4. No modifica el archivo original. Solo actúa sobre el DataFrame
   cargado en memoria.

Este módulo es consumido por:
- src/main.py
- src/nn/*
- src/core/config.py
- src/report/*
------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from src.core.utils import normalize_colnames


# ============================================================
# === FUNCIÓN PRIVADA: detección del bloque tabular ==========
# ============================================================

def _detect_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica la región válida dentro del DataFrame:

    Flujo:
        1. Se eliminan columnas completamente vacías.
        2. Se detecta la primera fila con algún valor no nulo.
        3. Esa fila se interpreta como encabezado real.
        4. Se eliminan filas vacías posteriores.
        5. Se normalizan nombres de columna (incluye símbolos lógicos).

    Retorna
    -------
    DataFrame limpio y con encabezados normalizados.
    """
    # 1. Eliminar columnas totalmente vacías
    df = df.dropna(axis=1, how="all")

    # 2. Encontrar la primera fila “válida”
    first_valid_row = None
    for idx, row in df.iterrows():
        if row.notna().any():
            first_valid_row = idx
            break

    if first_valid_row is None:
        raise ValueError("No se detectaron datos válidos en el archivo.")

    # 3. Usar esa fila como encabezado
    new_header = df.iloc[first_valid_row].astype(str)
    df = df.iloc[first_valid_row + 1:].copy()
    df.columns = new_header

    # 4. Eliminar filas vacías restantes
    df = df.dropna(how="all")

    # 5. Normalizar encabezados
    df.columns = normalize_colnames(df.columns)

    # Garantizar que todo quede como string inicialmente
    df = df.astype(str)

    return df.reset_index(drop=True)


# ============================================================
# === FUNCIÓN PÚBLICA: carga de dataset =======================
# ============================================================

def load_dataset(path: str, sheet: str | None = None) -> pd.DataFrame:
    """
    Carga un dataset desde un archivo .csv, .xlsx o .ods.

    Parámetros
    ----------
    path : str
        Ruta absoluta o relativa al dataset.
    sheet : str | None
        Nombre o índice de hoja (solo aplica para Excel/ODS).

    Retorna
    -------
    pd.DataFrame
        DataFrame limpio, con encabezados normalizados
        (incluyendo soporte para símbolos lógicos).
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    ext = p.suffix.lower()

    # --- Carga según formato ---
    if ext == ".csv":
        df = pd.read_csv(p, encoding="utf-8", header=None)

    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(p, sheet_name=sheet or 0,
                           engine="openpyxl", header=None)

    elif ext == ".ods":
        df = pd.read_excel(p, sheet_name=sheet or 0,
                           engine="odf", header=None)

    else:
        raise ValueError(f"Formato no soportado: {ext}")

    # --- Detección automática del bloque tabular ---
    df = _detect_table(df)

    return df

