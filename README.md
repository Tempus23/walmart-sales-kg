# 📈 Pronóstico de Demanda para Retail (Walmart Sales Forecasting)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

Proyecto de Machine Learning para predecir la demanda semanal de ventas en tiendas Walmart, utilizando múltiples algoritmos de Machine Learning y técnicas avanzadas de Ingeniería de Características para series temporales.

## 🎯 El Problema de Negocio

En el sector *retail*, un pronóstico de demanda impreciso genera dos problemas costosos:

1.  **Exceso de Stock (Overstock):** Aumento de costes de almacenamiento, capital inmovilizado y, en productos frescos, un incremento directo en el **desperdicio** (merma).
2.  **Rotura de Stock (Stockout):** Pérdida de ventas directas, insatisfacción del cliente y potencial fuga a la competencia.

Este proyecto construye modelos que pronostican la demanda futura para optimizar el inventario, reducir la merma y asegurar la disponibilidad del producto.

## 💾 Dataset

Se utilizó el dataset [Walmart Sales](https://www.kaggle.com/datasets/mikhail1681/walmart-sales) de Kaggle.

* **Periodo:** Datos históricos de ventas semanales.
* **Volumen:** ~6,435 registros.
* **Alcance:** Ventas semanales para 45 tiendas Walmart diferentes.
* **Variables:** Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment.

## 🛠️ Stack Tecnológico

- **Python 3.11**
- **Machine Learning:** LightGBM, XGBoost, CatBoost, Random Forest, Prophet, LSTM (TensorFlow)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **AutoML:** TPOT
- **API:** Kaggle API

## 📁 Estructura del Proyecto

```
walmart-sales-kg/
├── demand_forecasting/          # Módulo principal
│   ├── trainers/               # Implementaciones de diferentes modelos
│   │   ├── base.py            # Clase base abstracta
│   │   ├── lightGBM.py        # LightGBM Trainer
│   │   ├── XGBoost.py         # XGBoost Trainer
│   │   ├── catboost.py        # CatBoost Trainer
│   │   ├── randomforest.py    # Random Forest Trainer
│   │   ├── prophet.py         # Prophet Trainer
│   │   └── neural_network.py  # LSTM Neural Network
│   ├── walmart_data.py        # Descarga y procesamiento de datos
│   ├── model.py               # Utilidades de modelo
│   └── plotting.py            # Funciones de visualización
├── data/                       # Datos (gitignored)
├── outputs/                    # Modelos y visualizaciones (gitignored)
│   ├── models/                # Modelos guardados
│   ├── plots/                 # Gráficos generados
│   └── automl_pipelines/      # Pipelines de TPOT
├── training.py                # Script principal de entrenamiento
├── autotrain.py              # AutoML con TPOT
├── notebook.ipynb            # Notebook exploratorio
└── requirements.txt          # Dependencias
```

## 🚀 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/Tempus23/walmart-sales-kg.git
cd walmart-sales-kg
```

2. Crea un entorno virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Configura tu API de Kaggle (opcional, si necesitas descargar datos):
   - Obtén tu `kaggle.json` desde [Kaggle Account Settings](https://www.kaggle.com/account)
   - Colócalo en `~/.kaggle/kaggle.json` (Linux/Mac) o `C:\Users\<user>\.kaggle\kaggle.json` (Windows)

## 💻 Uso

### Entrenamiento de Modelos

Ejecuta el script principal de entrenamiento:

```bash
python training.py
```

Este script:
- Carga y procesa los datos automáticamente
- Genera características de ingeniería
- Entrena el modelo seleccionado (por defecto LSTM)
- Evalúa el rendimiento con MAE y WMAE
- Guarda el modelo en `outputs/models/`
- Genera visualizaciones en `outputs/plots/`

### AutoML con TPOT

Para búsqueda automática del mejor modelo:

```bash
python autotrain.py
```

### Cambiar de Modelo

En `training.py`, cambia el trainer en la línea 59:

```python
# Opciones disponibles:
trainer = LightGBMTrainer()
trainer = XGBoostTrainer()
trainer = CatBoostTrainer()
trainer = RandomForestTrainer()
trainer = ProphetTrainer()
trainer = LSTMTrainer()
```

## 🔬 Ingeniería de Características

El proyecto implementa las siguientes características:

### Características Temporales
- **Month, Quarter, Year, WeekOfYear:** Componentes temporales básicos
- **Características Cíclicas:** `MonthSin`, `MonthCos`, `WeekSin`, `WeekCos` para capturar estacionalidad

### Características de Lag (Rezagos)
- **ventas_lag_1:** Ventas de la semana anterior
- **ventas_lag_4:** Ventas de hace 4 semanas (~1 mes)
- **ventas_lag_52:** Ventas de hace 52 semanas (~1 año)

### Medias Móviles
- **media_movil_4_semanas:** Media de las últimas 4 semanas
- **media_movil_12_semanas:** Media de las últimas 12 semanas

### Variables Externas
- Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment (del dataset original)

## 🤖 Modelos Implementados

Todos los modelos heredan de `BaseModel` con interfaz común:

| Modelo | Tipo | Características |
|--------|------|----------------|
| **LightGBM** | Gradient Boosting | Rápido, eficiente, maneja categóricas |
| **XGBoost** | Gradient Boosting | Robusto, regularización avanzada |
| **CatBoost** | Gradient Boosting | Especializado en variables categóricas |
| **Random Forest** | Ensemble | Menos prone a overfitting |
| **Prophet** | Time Series | Diseñado específicamente para series temporales |
| **LSTM** | Deep Learning | Red neuronal recurrente para secuencias |

## 📊 Resultados

Los modelos se evalúan con:
- **MAE (Mean Absolute Error):** Error promedio en unidades de ventas
- **WMAE (Weighted MAE):** MAE ponderado por importancia

### Visualizaciones Generadas

- `importancia_caracteristicas.png`: Importancia de cada feature en el modelo
- `predicciones_tienda_1.png`: Comparación predicciones vs valores reales

## 📈 Próximos Pasos

- [ ] Implementar validación cruzada temporal
- [ ] Optimización de hiperparámetros con Optuna
- [ ] Ensemble de múltiples modelos
- [ ] Deploy con API REST (FastAPI)

## 👤 Autor

**Carlos Hernández Martínez**

- GitHub: [@Tempus23](https://github.com/Tempus23)

## 📄 Licencia

Este proyecto está bajo la licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
