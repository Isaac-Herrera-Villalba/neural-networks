# Proyecto: Neural Networks (Python 3.12+)

Implementación desde cero de los modelos clásicos de **Redes Neuronales**
vistos en la práctica:

- **Perceptrón** (clasificación lineal)
- **Regla Delta (ADALINE)** — aprendizaje lineal por MSE
- **Backpropagation** para MLP (1 capa oculta)

El sistema genera automáticamente un **reporte matemático en LaTeX → PDF**
con todos los cálculos paso a paso: net, salidas, errores, deltas, actualizaciones
de pesos y MSE por época, siguiendo estrictamente el método del PDF proporcionado.

---

## input.txt — Formato general

Cada bloque representa un experimento independiente.  
Los comentarios de línea (`#`) y de bloque (`/* ... */`) son preservados.

**Campos principales:**

| Clave | Descripción |
|------|-------------|
| `DATASET` | Archivo CSV / XLSX / ODS |
| `SHEET` | Hoja dentro del archivo (solo para Excel/ODS) |
| `METHOD` | PERCEPTRON / DELTA / BACKPROP |
| `X_COLS` | Lista de entradas (X₁, X₂, …) |
| `Y_COL` | Columna objetivo |
| `LEARNING_RATE` | η |
| `MAX_EPOCHS` | Número máximo de épocas |
| `ERROR_THRESHOLD` | Solo para Delta/Backprop |
| `HIDDEN_NEURONS` | Solo para Backprop |
| `ACTIVATION` | Función (SIGMOID por defecto) |
| `REPORT` | Ruta del PDF final |

Puedes incluir **varios bloques** con diferentes métodos y datasets.

---

## Ejemplo ÚNICO (MLP Backpropagation – XOR)

```txt
/* Ejemplo: Backpropagation para XOR */

DATASET = data/xor.ods
SHEET = XOR
METHOD = BACKPROP

X_COLS = X1, X2
Y_COL = Y

LEARNING_RATE = 0.5
HIDDEN_NEURONS = 2
MAX_EPOCHS = 200
ERROR_THRESHOLD = 0.01
ACTIVATION = SIGMOID

REPORT = output/reporte_nn.pdf

