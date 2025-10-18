# 📈 Pronóstico de Demanda para Retail (Store Item Demand)

Proyecto de Machine Learning para predecir la demanda diaria de artículos (nivel SKU) en 10 tiendas diferentes, utilizando un enfoque de Gradient Boosting (LightGBM) e Ingeniería de Características de series temporales.

El notebook principal (`notebooks/pronostico_demanda.ipynb`) es 100% verificable y ejecutable en Google Colab.

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
