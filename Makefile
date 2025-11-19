# Makefile
# -----------------------------------------------------------------------
# Descripción:
# Automatiza el flujo completo del proyecto **Neural Networks**.
#
# Métodos soportados:
#   - Perceptrón
#   - Regla Delta (ADALINE)
#   - Backpropagation (MLP)
#
# Flujo FULL (make full):
#   1. Ejecutar input.txt (genera .tex desde src/main.py)
#   2. Compilar LaTeX → PDF
#   3. Abrir el PDF automáticamente
# -----------------------------------------------------------------------

PYTHON         = python3
SRC_DIR        = src
OUT_DIR        = output
INPUT_FILE     = input.txt
PDF_READER     = okular
PYTHON_MODULES = __pycache__

LATEX_REPORT_NAME = reporte_nn

# Módulo principal (para 'python -m'):
MAIN_MODULE   = src.main

VENV_DIR      = .venv
PYTHON_VENV   = $(VENV_DIR)/bin/python
PIP_VENV      = $(VENV_DIR)/bin/pip

PDF_FILE      = $(OUT_DIR)/$(LATEX_REPORT_NAME).pdf
TEX_FILE      = $(OUT_DIR)/$(LATEX_REPORT_NAME).tex

.PHONY: all help env run latex pdf view full clean

# -----------------------------------------------------------------------
all: help

help:
	@echo "Comandos disponibles:"
	@echo "  make env    -> Crea entorno virtual e instala dependencias (se usa solo 1 vez)"
	@echo "  make run    -> Ejecuta el sistema de Redes Neuronales (genera .tex y PDF)"
	@echo "  make latex  -> Compila manualmente el archivo .tex (si quieres recompilar)"
	@echo "  make pdf    -> Alias de 'make run'"
	@echo "  make view   -> Abre el PDF generado"
	@echo "  make full   -> Ejecuta run + latex + view (flujo COMPLETO, sin env)"
	@echo "  make clean  -> Elimina archivos temporales"
	@echo "---------------------------------------------------------------"

# -----------------------------------------------------------------------
env:
	@echo "\n=== Creando entorno virtual (.venv) ===\n"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "\n[OK] Entorno virtual creado\n"; \
	else \
		echo "\n[INFO] Ya existe .venv\n"; \
	fi
	@echo "\n=== Instalando dependencias ===\n"
	$(PIP_VENV) install --upgrade pip >/dev/null
	$(PIP_VENV) install -r requirements.txt
	@echo "\n[OK] Dependencias instaladas\n"

# -----------------------------------------------------------------------
run:
	@echo "\n=== Ejecutando redes neuronales ===\n"
	mkdir -pv $(OUT_DIR)/
	$(PYTHON_VENV) -m $(MAIN_MODULE)
	@echo "\n[OK] Ejecución completada.\n"

pdf: run

# -----------------------------------------------------------------------
latex:
	@echo "\n=== Compilando LaTeX manualmente con pdflatex ==="
	@if [ -f "$(TEX_FILE)" ]; then \
		cd $(OUT_DIR) && \
		pdflatex -interaction=nonstopmode $(notdir $(TEX_FILE)) >/dev/null 2>&1; \
		echo "\n[OK] PDF recompilado: $(PDF_FILE)\n"; \
	else \
		echo "\n[ERROR] No se encontró $(TEX_FILE). Ejecuta primero 'make run'. \n"; \
	fi

# -----------------------------------------------------------------------
view:
	@echo "\n=== Abriendo PDF con Okular ===\n"
	$(PDF_READER) $(PDF_FILE) &

# -----------------------------------------------------------------------
# FULL: ejecuta TODO el proceso (sin env)
# -----------------------------------------------------------------------
full:
	@echo "\n=== EJECUCIÓN COMPLETA DEL PROYECTO ===\n"
	make run
	make latex
	make view
	@echo "\n=== FLUJO COMPLETO FINALIZADO ===\n"

# -----------------------------------------------------------------------
clean:
	@echo "\nEliminando archivos generados...\n"
	rm -rf $(OUT_DIR)/
	find $(SRC_DIR) -type d -name $(PYTHON_MODULES) -exec rm -rf {} +
	rm -f data/.~lock.*
	@echo "\n[OK] Limpieza completada.\n"

