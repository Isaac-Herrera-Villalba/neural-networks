# Makefile
# -----------------------------------------------------------------------
# Descripción:
# Automatiza el flujo completo del proyecto **Neural Networks**.
#
# Este proyecto implementa los siguientes métodos clásicos de redes
# neuronales (según la práctica y el PDF proporcionado):
#
#   - Perceptrón simple
#   - Regla Delta (ADALINE)
#   - Backpropagation (MLP)
#
# El sistema toma la configuración definida en `input.txt`,
# carga datasets (.csv, .xlsx, .ods), entrena la red seleccionada
# y genera automáticamente un **reporte en PDF** con todos los
# cálculos paso a paso.
#
# Flujo general:
#   1. Crear entorno virtual e instalar dependencias    → make env
#   2. Ejecutar los experimentos definidos en input.txt → make run
#   3. Compilar manualmente LaTeX (opcional)            → make latex
#   4. Abrir el PDF final                               → make view
#
# Dependencias administradas en requirements.txt.
# -----------------------------------------------------------------------

PYTHON         = python3
SRC_DIR        = src
OUT_DIR        = output
INPUT_FILE     = input.txt
MAIN_FILE      = $(SRC_DIR)/main.py
PDF_READER     = okular
PYTHON_MODULES = __pycache__

# El PDF final es definido por el último bloque con REPORT= en input.txt
# Para propósitos de compilación manual, se usa un nombre genérico:
LATEX_REPORT_NAME = reporte_nn

VENV_DIR      = .venv
PYTHON_VENV   = $(VENV_DIR)/bin/python
PIP_VENV      = $(VENV_DIR)/bin/pip

PDF_FILE      = $(OUT_DIR)/$(LATEX_REPORT_NAME).pdf
TEX_FILE      = $(OUT_DIR)/$(LATEX_REPORT_NAME).tex

.PHONY: all help env run latex pdf view clean full

# -----------------------------------------------------------------------
all: help

help:
	@echo "Comandos disponibles:"
	@echo "  make env    -> Crea entorno virtual e instala dependencias"
	@echo "  make run    -> Ejecuta el sistema de Redes Neuronales (genera PDF)"
	@echo "  make latex  -> Compila manualmente el archivo .tex con LaTeX"
	@echo "  make pdf    -> Alias de 'make run'"
	@echo "  make view   -> Abre el PDF generado"
	@echo "  make clean  -> Elimina archivos temporales y auxiliares"
	@echo "  make full   -> Ejecuta env + run + view"
	@echo "---------------------------------------------------------------"

# -----------------------------------------------------------------------
env:
	@echo "=== Creando entorno virtual (.venv) ==="
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "[OK] Entorno virtual creado"; \
	else \
		echo "[INFO] Ya existe .venv"; \
	fi
	@echo "=== Instalando dependencias ==="
	$(PIP_VENV) install --upgrade pip >/dev/null
	$(PIP_VENV) install -r requirements.txt
	@echo "[OK] Dependencias instaladas"

# -----------------------------------------------------------------------
run:
	@echo "=== Ejecutando entrenamiento de Redes Neuronales ==="
	mkdir -pv $(OUT_DIR)/
	$(PYTHON_VENV) $(MAIN_FILE)
	@echo "[OK] Ejecución completada."

pdf: run

# -----------------------------------------------------------------------
latex:
	@echo "=== Compilando LaTeX manualmente con pdflatex ==="
	@if [ -f "$(TEX_FILE)" ]; then \
		cd $(OUT_DIR) && \
		pdflatex -interaction=nonstopmode $(notdir $(TEX_FILE)) >/dev/null 2>&1; \
		echo "[OK] PDF recompilado: $(PDF_FILE)"; \
	else \
		echo "[ERROR] No se encontró $(TEX_FILE). Ejecuta primero 'make run'."; \
	fi

# -----------------------------------------------------------------------
view:
	@echo "Abriendo PDF con Okular..."
	$(PDF_READER) $(PDF_FILE) &

# -----------------------------------------------------------------------
full: env run view

# -----------------------------------------------------------------------
clean:
	@echo "Eliminando archivos generados..."
	rm -rf $(OUT_DIR)/
	find $(SRC_DIR) -type d -name $(PYTHON_MODULES) -exec rm -rf {} +
	rm -f data/.~lock.*
	@echo "[OK] Limpieza completada."

