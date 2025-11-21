# Neural Networks – Perceptrón Simple (Python 3.12+)

Implementación desde cero del **Perceptrón Simple de Rosenblatt**, siguiendo el
material teórico *“Learning Perceptrons (diapositivas 1–8)”*.

El proyecto permite ejecutar múltiples experimentos definidos en `input.txt`.
Cada experimento entrena un perceptrón independiente y genera un apartado
propio dentro del reporte final **LaTeX → PDF**, incluyendo:

- Definición matemática del perceptrón  
- Regla de aprendizaje  
- Algoritmo en pseudocódigo  
- Vista previa del dataset  
- Cálculo del hiperplano aprendido  
- Historial de pesos  
- Historial de errores por época  
- Análisis de convergencia / no-convergencia  
- Conclusiones formales del experimento  

El objetivo es analizar funciones lógicas y datasets con diferente número de
variables y filas, incluyendo casos **linealmente separables y no separables**.

---

## `input.txt` — Formato general

Cada bloque representa un experimento independiente.  
Los comentarios de línea (`#`, `//`) y de bloque (`/* … */`) son aceptados.

**Campos principales:**

| Clave            | Descripción                               |
|------------------|--------------------------------------------|
| `DATASET`        | Archivo CSV / XLSX / ODS                   |
| `SHEET`          | Hoja dentro del archivo (solo Excel/ODS)   |
| `X_COLS`         | Columnas de entrada                        |
| `Y_COL`          | Columna objetivo                           |
| `LEARNING_RATE`  | η                                          |
| `MAX_EPOCHS`     | Número máximo de épocas                    |
| `INITIAL_WEIGHTS`| Pesos iniciales opcionales                 |

Puedes definir **varios bloques** con distintos datasets, hojas y parámetros.
Cada uno generará su propia sección dentro del PDF final.

---

## Ejemplo básico de bloque (`input.txt`)

```txt
/* Perceptrón sobre tabla AND (dataset lógico simple) */

NN = PERCEPTRON
DATASET = data/dataset.ods
SHEET = Sheet1

X_COLS = x1, x2
Y_COL  = AND

LEARNING_RATE = 0.5
MAX_EPOCHS = 30

