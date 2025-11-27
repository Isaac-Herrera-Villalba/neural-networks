#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/core/data_extractor/loader.py
---------------------------------------------------------------------------
Módulo: Carga estructurada y normalización de datasets
---------------------------------------------------------------------------

Finalidad general
-----------------
Este módulo es el responsable de cargar datasets provenientes de archivos
tabulares de distintos formatos:

    - CSV  (.csv)
    - Excel (.xlsx / .xls)
    - OpenDocument Spreadsheet (.ods)

La funcionalidad principal consiste en extraer la tabla “real” incluso
si el archivo contiene irregularidades típicas como:

    • Filas vacías al inicio
    • Columnas vacías o separadores adicionales
    • Encabezados desplazados hacia abajo
    • Celdas faltantes o ruidosas

Posteriormente, se normalizan los encabezados mediante:

    normalize_colnames()  → nombres ASCII limpios
    normalize_logic_symbol() → soporte para símbolos lógicos

de forma que columnas como:

        AND, OR, XOR, ↔, ⊕, ∧, <=>, ...

queden mapeadas consistentemente en el DataFrame.

Este módulo es fundamental para el flujo completo, ya que todos los
modelos de redes neuronales (perceptrón, etc.) dependen de obtener un
DataFrame completamente limpio y normalizado.

Se usa en:
    - src/main.py
    - src/nn/perceptron.py
    - src/core/config.py
    - src/report/*
---------------------------------------------------------------------------

"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from src.core.utils import normalize_colnames


# ============================================================================
# === FUNCIÓN PRIVADA: detección automática del bloque tabular válido =======
# ============================================================================

def _detect_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta y extrae la tabla válida dentro del DataFrame cargado desde
    un archivo externo.

    Esta función es necesaria porque muchos archivos .ods/.xlsx contienen
    filas o columnas vacías, o un encabezado que no está en la fila 0.

    Flujo interno:
    --------------
    1. Se eliminan columnas completamente vacías.
    2. Se identifica la primera fila que contiene al menos un dato válido.
       → Se asume como encabezado real.
    3. Se desplaza el DataFrame para usar esa fila como encabezado.
    4. Se eliminan filas totalmente vacías que puedan haber quedado.
    5. Normaliza los nombres de columna:
            - elimina espacios
            - convierte símbolos lógicos a nombres ASCII seguros
    6. Convierte las celdas a string para evitar problemas en etapas
       posteriores donde se mezclan tipos numéricos y texto.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame tal como se cargó desde el archivo.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpio, con encabezados correctos y filas válidas.
    """

    # 1. Eliminar columnas completamente vacías
    df = df.dropna(axis=1, how="all")

    # 2. Encontrar la primera fila con al menos un valor no vacío
    first_valid_row = None
    for idx, row in df.iterrows():
        if row.notna().any():
            first_valid_row = idx
            break

    if first_valid_row is None:
        raise ValueError("No se detectaron datos válidos en el archivo.")

    # 3. La fila encontrada se usa como encabezado real
    new_header = df.iloc[first_valid_row].astype(str)
    df = df.iloc[first_valid_row + 1:].copy()
    df.columns = new_header

    # 4. Eliminar filas totalmente vacías restantes
    df = df.dropna(how="all")

    # 5. Normalización automática de encabezados
    df.columns = normalize_colnames(df.columns)

    # 6. Convertir todo a string
    df = df.astype(str)

    return df.reset_index(drop=True)


# ============================================================================
# === FUNCIÓN PÚBLICA: interface de carga de dataset =========================
# ============================================================================

def load_dataset(path: str, sheet: str | None = None) -> pd.DataFrame:
    """
    Carga un dataset desde un archivo .csv, .xlsx o .ods, aplica limpieza
    estructural y normaliza nombres de columna.

    Esta es la función que debe usarse externamente por todo el sistema.
    Automatiza la lectura correcta del archivo, sin que el usuario tenga
    que preocuparse por encabezados mal colocados o celdas vacías.

    Parámetros
    ----------
    path : str
        Ruta del archivo tabular.
    sheet : str | None
        Nombre de la hoja (solo aplica para Excel y ODS).
        Si no se especifica, se cargará la primera hoja.

    Retorna
    -------
    pd.DataFrame
        DataFrame procesado y normalizado, apto para uso en los módulos
        de redes neuronales.

    Errores posibles
    ----------------
    FileNotFoundError : si el archivo no existe.
    ValueError        : si el formato no es soportado o no se pudo
                        detectar una tabla válida.
    """

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    ext = p.suffix.lower()

    # --- Carga bruta según tipo de archivo ---
    if ext == ".csv":
        df = pd.read_csv(p, encoding="utf-8", header=None)

    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(
            p,
            sheet_name=sheet or 0,
            engine="openpyxl",
            header=None
        )

    elif ext == ".ods":
        df = pd.read_excel(
            p,
            sheet_name=sheet or 0,
            engine="odf",
            header=None
        )

    else:
        raise ValueError(f"Formato no soportado: {ext}")

    # --- Detección automática del bloque tabular válido ---
    df = _detect_table(df)

    return df

