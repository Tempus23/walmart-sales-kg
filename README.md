# 📈 Pronóstico de Demanda para Retail (Store Item Demand)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Demand%20Forecasting-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

Proyecto de Machine Learning para predecir la demanda diaria de artículos (nivel SKU) en 10 tiendas diferentes, utilizando un enfoque de Gradient Boosting (LightGBM) e Ingeniería de Características de series temporales.

El notebook principal (`notebooks/notebook.ipynb`) es 100% verificable y ejecutable en Google Colab.

## 🎯 El Problema de Negocio

En el sector *retail*, un pronóstico de demanda impreciso genera dos problemas costosos:

1.  **Exceso de Stock (Overstock):** Aumento de costes de almacenamiento, capital inmovilizado y, en productos frescos (como en Mercadona), un incremento directo en el **desperdicio** (merma).
2.  **Rotura de Stock (Stockout):** Pérdida de ventas directas, insatisfacción del cliente y potencial fuga a la competencia.

Este proyecto construye un modelo que pronostica la demanda futura para optimizar el inventario, reducir la merma y asegurar la disponibilidad del producto.

## 💾 Dataset

Se utilizó el dataset [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/store-item-demand-forecasting-challenge) de Kaggle.

* **Periodo:** 5 años de datos (2013 - 2017).
* **Volumen:** ~913,000 registros.
* **Alcance:** Ventas diarias para 50 artículos (SKUs) en 10 tiendas diferentes.

## 🚀 Características Principales

- ✅ **Múltiples Modelos**: Implementación de 6 algoritmos diferentes (LightGBM, XGBoost, CatBoost, Random Forest, Prophet, LSTM)
- ✅ **Ingeniería de Características**: Features avanzadas de series temporales (lags, medias móviles, codificación cíclica)
- ✅ **AutoML**: Búsqueda automática de pipelines con TPOT
- ✅ **Visualizaciones**: Gráficos de predicciones e importancia de características
- ✅ **Arquitectura Modular**: Código organizado y extensible

## 🛠️ Stack Tecnológico

### Lenguaje
- **Python 3.12**

### Librerías de Machine Learning
- **LightGBM**: Gradient Boosting principal
- **XGBoost**: Gradient Boosting alternativo
- **CatBoost**: Boosting optimizado para categóricas
- **Scikit-learn**: Métricas y utilidades ML
- **Prophet**: Modelo de series temporales de Facebook
- **TensorFlow**: Redes neuronales (LSTM)
- **TPOT**: AutoML para optimización de pipelines

### Análisis de Datos y Visualización
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas
- **Matplotlib**: Visualizaciones
- **Seaborn**: Visualizaciones estadísticas

### Otros
- **Kaggle**: API para descarga de datos

## 📦 Instalación

### Prerrequisitos
- Python 3.12 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/Tempus23/walmart-sales-kg.git
cd walmart-sales-kg
```

2. **Crear un entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar Kaggle API (para descargar datos)**

Descarga tu archivo `kaggle.json` desde [Kaggle Account Settings](https://www.kaggle.com/account) y colócalo en:
- Linux/Mac: `~/.kaggle/kaggle.json`
- Windows: `C:\Users\<Usuario>\.kaggle\kaggle.json`

Asegúrate de que tiene los permisos correctos:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

## 💻 Uso

### Entrenar un Modelo

Para entrenar un modelo con LightGBM (o cambiar a otro modelo como XGBoost, CatBoost, etc.):

```bash
python training.py
```

El script `training.py` realiza:
1. Carga y preprocesamiento de datos
2. Creación de características de series temporales
3. División de datos en entrenamiento/validación
4. Entrenamiento del modelo
5. Evaluación y cálculo de métricas
6. Guardado del modelo entrenado
7. Generación de visualizaciones

### AutoML con TPOT

Para buscar automáticamente el mejor pipeline:

```bash
python autotrain.py
```

Este proceso puede tardar considerable tiempo dependiendo de la configuración de `generations` y `population_size`.

### Explorar el Notebook

El notebook Jupyter con análisis exploratorio y experimentación está disponible en:

```bash
jupyter notebook notebooks/notebook.ipynb
```

## 📁 Estructura del Proyecto

```
walmart-sales-kg/
├── demand_forecasting/          # Paquete principal
│   ├── __init__.py
│   ├── model.py                 # Funciones de modelo y decoradores
│   ├── walmart_data.py          # Cargador de datos y feature engineering
│   ├── plotting.py              # Funciones de visualización
│   └── trainers/                # Implementaciones de modelos
│       ├── __init__.py
│       ├── base.py              # Clase base abstracta
│       ├── lightGBM.py          # Modelo LightGBM
│       ├── XGBoost.py           # Modelo XGBoost
│       ├── catboost.py          # Modelo CatBoost
│       ├── randomforest.py      # Modelo Random Forest
│       ├── prophet.py           # Modelo Prophet
│       └── neural_network.py    # Modelo LSTM
├── notebooks/                   # Jupyter notebooks
│   └── notebook.ipynb           # Notebook principal
├── data/                        # Datos (no incluidos en repo)
├── outputs/                     # Salidas del entrenamiento
│   ├── models/                  # Modelos guardados
│   ├── plots/                   # Visualizaciones
│   └── automl_pipelines/        # Pipelines de TPOT
├── training.py                  # Script de entrenamiento principal
├── autotrain.py                 # Script de AutoML
├── requirements.txt             # Dependencias del proyecto
├── README.md                    # Este archivo
├── RESULTS.md                   # Resultados y métricas detalladas
└── LICENSE                      # Licencia MIT
```

## 📊 Resultados

Para ver resultados detallados, métricas y análisis de los modelos, consulta el archivo [RESULTS.md](RESULTS.md).

### Métricas Principales
- **MAE (Mean Absolute Error)**: Error absoluto medio
- **WMAE (Weighted MAE)**: Error absoluto medio ponderado

## 🔧 Configuración Avanzada

### Cambiar el Modelo

En `training.py`, modifica la línea donde se instancia el trainer:

```python
# Cambiar de LSTMTrainer a otro modelo:
trainer = LightGBMTrainer()    # LightGBM
# trainer = XGBoostTrainer()   # XGBoost
# trainer = CatBoostTrainer()  # CatBoost
# trainer = RandomForestTrainer()  # Random Forest
# trainer = ProphetTrainer()   # Prophet
```

### Ajustar Características

Edita las listas `CARACTERISTICAS` y `CARACTERISTICAS_CATEGORICAS` en `training.py` para experimentar con diferentes conjuntos de features.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 👤 Autor

**Carlos Hernández Martínez**

## 🙏 Agradecimientos

- Dataset proporcionado por [Kaggle](https://www.kaggle.com/c/store-item-demand-forecasting-challenge)
- Inspiración en casos de uso reales del sector retail

---

⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!
