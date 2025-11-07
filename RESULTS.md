# 📊 Resultados del Proyecto

## Objetivo del Modelo

El objetivo principal de este proyecto es predecir la demanda diaria de artículos (nivel SKU) en 10 tiendas diferentes utilizando técnicas de Machine Learning y análisis de series temporales.

## Métricas de Evaluación

Para evaluar el rendimiento de los modelos, utilizamos las siguientes métricas:

- **MAE (Mean Absolute Error)**: Error absoluto medio entre las predicciones y los valores reales
- **WMAE (Weighted Mean Absolute Error)**: Error absoluto medio ponderado, que da más importancia a ciertos productos o tiendas

## Modelos Implementados

El proyecto incluye implementaciones de múltiples algoritmos de Machine Learning:

### 1. LightGBM (Gradient Boosting)
- **Descripción**: Modelo principal basado en Gradient Boosting optimizado para velocidad y eficiencia
- **Características**: Ingeniería de características de series temporales incluyendo lags, medias móviles y codificación cíclica

### 2. XGBoost
- **Descripción**: Implementación alternativa de Gradient Boosting
- **Ventajas**: Alto rendimiento y robustez

### 3. CatBoost
- **Descripción**: Gradient Boosting optimizado para variables categóricas
- **Ventajas**: Manejo nativo de características categóricas

### 4. Random Forest
- **Descripción**: Ensemble de árboles de decisión
- **Ventajas**: Interpretabilidad y resistencia al overfitting

### 5. Prophet
- **Descripción**: Modelo de series temporales de Facebook
- **Ventajas**: Manejo automático de estacionalidad y tendencias

### 6. LSTM (Neural Network)
- **Descripción**: Red neuronal recurrente para series temporales
- **Ventajas**: Capacidad para capturar patrones complejos y dependencias a largo plazo

## Características del Modelo

El modelo utiliza las siguientes características para realizar predicciones:

### Características Temporales
- `Store`: Identificador de la tienda
- `Holiday_Flag`: Indicador de días festivos
- `Month`, `Quarter`, `Year`: Componentes de fecha
- `Week_of_Year`: Semana del año

### Características Cíclicas
- `Month_sin`, `Month_cos`: Codificación cíclica del mes
- `Week_sin`, `Week_cos`: Codificación cíclica de la semana

### Características de Series Temporales
- `Ventas_Lag_1`: Ventas de la semana anterior
- `Ventas_Lag_4`: Ventas de hace 4 semanas
- `Ventas_Lag_52`: Ventas del año anterior (estacionalidad anual)
- `Media_Movil_4_Semanas`: Media móvil de 4 semanas
- `Media_Movil_12_Semanas`: Media móvil de 12 semanas

## Dataset

- **Fuente**: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/store-item-demand-forecasting-challenge) de Kaggle
- **Periodo**: 5 años de datos (2013 - 2017)
- **Volumen**: ~913,000 registros
- **Alcance**: Ventas diarias para 50 artículos (SKUs) en 10 tiendas diferentes

## División de Datos

- **Fecha de corte**: 2012-05-01
- **Conjunto de entrenamiento**: Datos anteriores a la fecha de corte
- **Conjunto de validación**: Datos posteriores a la fecha de corte

## AutoML con TPOT

El proyecto también incluye una implementación de AutoML utilizando TPOT (Tree-based Pipeline Optimization Tool) que:
- Busca automáticamente el mejor pipeline de preprocessing y modelo
- Optimiza hiperparámetros
- Genera código Python del mejor pipeline encontrado

## Visualizaciones

El proyecto genera las siguientes visualizaciones:

1. **Predicciones vs Valores Reales**: Comparación visual de las predicciones del modelo con los valores reales para tiendas específicas
2. **Importancia de Características**: Gráfico que muestra qué características son más importantes para las predicciones del modelo

## Archivos de Salida

Los resultados del entrenamiento se guardan en:
- `outputs/models/`: Modelos entrenados
- `outputs/plots/`: Visualizaciones generadas
- `outputs/automl_pipelines/`: Pipelines generados por TPOT

## Próximos Pasos

Para mejorar aún más el modelo, se podrían explorar:
- Incorporación de datos externos (clima, eventos especiales, promociones)
- Ensemble de múltiples modelos
- Ajuste fino de hiperparámetros con búsqueda bayesiana
- Implementación de modelos de deep learning más avanzados (Transformers)
- Predicción a múltiples horizontes temporales
