
# 🧠 GCN sobre el KarateClub Dataset

Este proyecto es una implementación de una **Red Neuronal Convolucional sobre Grafos (GCN)** utilizando el dataset clásico **KarateClub** de Wayne W. Zachary. El objetivo es clasificar los nodos (personas) de un grafo en dos comunidades principales, aplicando técnicas de machine learning sobre grafos mediante [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/). Cabe recalcar que, se necesitó asistencia de la inteligencia artificial (ChatGPT por OpenAI) para poder llevar a cabo parte del código, pero los conceptos fundamentales y la idea de qué se quería realizar fue estudiada y moldeada por los integrantes del grupo en conjunto.

Esta demo correspondiente al curso **Estructuras Discretas para Informática (EIF203)** impartido por **Dr. Carlos Loría-Sáenz**.

---

## 🛠️ Tecnologías Utilizadas

- **Python ≥ 3.8**
- **PyTorch** – Framework para Deep Learning.
- **PyTorch Geometric** – Biblioteca para redes neuronales sobre grafos.
- **NetworkX** – Librería para la manipulación y análisis de grafos.
- **Matplotlib** – Librería para la visualización de datos.

---

## 🦾 Implementación

El modelo se basa en una arquitectura de **GCN** (Graph Convolutional Network), que posee dos capas de convolución para redes neuronales sobre grafos (`GCNConv`) y una función de activación ReLU.

### 📊 ¿Cómo funciona?

1. **Dataset**: Se utiliza el **KarateClub**, que representa las relaciones sociales en un club de karate.
2. **Preprocesamiento**: Las etiquetas se convierten a dos clases: LDA y CSH.
3. **Entrenamiento**: El modelo se entrena con `CrossEntropyLoss` y el optimizador `Adam`.
4. **Evaluación**: Se predicen las clases de los nodos y se visualiza la comparación con las etiquetas reales.

---

## 📁 Estructura del Proyecto

```
gcn_demo/
├── config.py              # Parámetros de entrenamiento y visualización
├── main.py                # Entrenamiento + evaluación automática
├── train.py               # Fase de entrenamiento del modelo GCN
├── evaluate.py            # Evaluación y visualización de resultados
├── requirements.txt       # Librerías necesarias
├── model/
│   └── gcn_model.py       # Implementación de la arquitectura GCN
├── utils/
│   └── visualize.py       # Función para graficar el grafo
└─
```

---

## 🧪 Requisitos

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

---

## ⚙️ ¿Cómo Ejecutarlo?

### Entrenamiento y evaluación conjunta:

```bash
python main.py
```

> Este comando entrena el modelo y luego muestra dos gráficas:
> - **Grafo real**: etiquetas verdaderas del dataset.
> - **Grafo predicho**: predicciones del modelo post-entrenamiento.

### Solamente el training:

```bash
python train.py
```

### Evaluación con modelo entrenado:

```bash
python evaluate.py
```

---

## 👥 Estudiantes detrás de la demo:

- Sebastián David González Masis – 119220546  
- Juan Pablo Sánchez Bermúdez – 119560437  
- Karen Minards Avilés López – 208160954
