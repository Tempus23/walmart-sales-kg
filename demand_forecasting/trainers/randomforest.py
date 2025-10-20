# demand_forecasting/random_forest_model.py
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
from .base import BaseModel
from ..walmart_data import WalmartFeatures


class RandomForestTrainer(BaseModel):
    def __init__(
        self,
        n_estimators=100,  # RF es más rápido, 100 es un buen baseline
        criterion="absolute_error",  # Para optimizar MAE
        n_jobs=-1,
        random_state=42,
        max_depth=10,  # Para evitar overfitting
    ) -> None:

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            n_jobs=n_jobs,
            random_state=random_state,
            max_depth=max_depth,
        )

    def train(self, X_train, y_train, X_val, y_val, caracteristicas_categoricas):
        """
        Entrena y devuelve un modelo RandomForest.
        Nota: RandomForest de Sklearn no usa eval_set ni early_stopping.
        """

        print("Iniciando entrenamiento de Random Forest...")
        # Convertir nombres de columnas a string para evitar errores de sklearn
        if hasattr(X_train, "columns"):
            X_train.columns = [str(col) for col in X_train.columns]
        if hasattr(X_val, "columns"):
            X_val.columns = [str(col) for col in X_val.columns]
        # Ignoramos X_val, y_val, y caracteristicas_categoricas
        # ya que .fit() no los utiliza para early stopping.
        self.model.fit(X_train, y_train)
        print("¡Entrenamiento completado!")
        return self.model

    def evaluate(self, X_val, y_val):
        """Calcula y muestra las métricas de error WMAE y MAE."""

        predicciones_val = self.model.predict(X_val)

        # Calcular WMAE (Métrica clave de Walmart)
        weights = X_val[WalmartFeatures.HOLIDAY_FLAG.value].apply(
            lambda x: 5 if x else 1
        )
        error_absoluto = np.abs(y_val - predicciones_val)
        wmae = (np.sum(weights * error_absoluto)) / (np.sum(weights))

        mae_normal = mean_absolute_error(
            np.asarray(y_val), np.asarray(predicciones_val)
        )

        print(f"======================================================")
        print(f"Error Absoluto Medio (MAE) Normal:    ${mae_normal:,.2f}")
        print(f"Error Absoluto Medio PONDERADO (WMAE): ${wmae:,.2f}")
        print(f"======================================================")

        return wmae, mae_normal

    def predict(self, X):
        """Genera predicciones utilizando el modelo entrenado."""
        return self.model.predict(X)

    def save(self, filepath: Path):
        """Guarda el modelo entrenado en un archivo."""
        print(f"Guardando modelo en: {filepath}")
        joblib.dump(self.model, filepath)

    def plot_feature_importance(self, save_path: Path = None):
        """Grafica la importancia de las características del modelo."""
        import matplotlib.pyplot as plt

        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_

        # Ordenar por importancia
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 6))
        plt.title("Importancia de las Características - Random Forest")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(
            range(len(importances)), [feature_names[i] for i in indices], rotation=90
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Importancia de características guardada en: {save_path}")
        plt.show()
