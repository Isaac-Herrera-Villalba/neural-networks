# Inteligencia Artificial, 2026-1
## Equipo 6

### Integrantes
· Fuentes Jiménez Yasbeck Dailanny  
· Herrera Villalba Isaac  
· Juárez García Yuliana  
· Ruiz Bastián Óscar  
· Sampayo Aguilar Cinthia Gloricel  
· Velarde Valencia Josue  

---

## Proyecto: Neural Networks

### Descripción general

Este proyecto implementa un sistema basado en **redes neuronales artificiales (ANNs)** diseñado para resolver problemas de **predicción, clasificación o regresión**, según el dataset utilizado.  
El objetivo es comprender el funcionamiento interno de una red neuronal —desde el procesamiento de datos hasta el entrenamiento y la generación de resultados—, aplicando fundamentos teóricos vistos en clase.

El sistema trabaja con datasets en formato **CSV**, **XLSX** o **ODS**, y permite ajustar diversos parámetros como:

- Número de capas y neuronas.  
- Tasa de aprendizaje (*learning rate*).  
- Función de activación.  
- Número de épocas de entrenamiento.  
- Porcentaje de división entre entrenamiento y prueba.  

Además, genera un **reporte en PDF** con los resultados obtenidos, incluyendo métricas de desempeño, gráficas de error y resumen del modelo.

---

### Funcionamiento

#### 1. Entrada

El archivo `input.txt` actúa como **fuente de configuración principal**.  
Define las características del dataset, la estructura de la red neuronal y los parámetros del entrenamiento.

##### Estructura general

| Clave | Descripción |
|-------|--------------|
| `DATASET` | Ruta del archivo de datos (`.ods`, `.xlsx`, `.csv`). |
| `SHEET` | *(Opcional)* Hoja a usar en caso de archivos de tipo hoja de cálculo. |
| `TARGET_COLUMN` | Columna objetivo (variable dependiente). |
| `HIDDEN_LAYERS` | Lista del número de neuronas por capa oculta (por ejemplo: `[8, 4]`). |
| `ACTIVATION` | Función de activación a utilizar (`sigmoid`, `relu`, `tanh`, etc.). |
| `LEARNING_RATE` | Tasa de aprendizaje del modelo (por defecto `0.01`). |
| `EPOCHS` | Número de iteraciones de entrenamiento. |
| `TEST_SPLIT` | Porcentaje de datos reservados para prueba (por defecto `0.2`). |
| `REPORT` | Ruta y nombre del archivo PDF generado con los resultados. |

##### Ejemplo de configuración activa

```txt
DATASET=data/ventas.ods
SHEET=Sheet1
TARGET_COLUMN=Ganancia
HIDDEN_LAYERS=[10, 5]
ACTIVATION=relu
LEARNING_RATE=0.01
EPOCHS=200
TEST_SPLIT=0.2
REPORT=output/reporte.pdf

