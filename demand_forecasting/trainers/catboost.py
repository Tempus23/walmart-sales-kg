# demand_forecasting/catboost_model.py
from catboost import CatBoostRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
from .base import BaseModel
from ..walmart_data import WalmartFeatures


class CatBoostTrainer(BaseModel):
    def __init__(
        self,
        loss_function="MAE",  # Equivalente a 'regression_l1'
        eval_metric="MAE",
        iterations=1000,
        learning_rate=0.01,
        thread_count=-1,
        random_seed=42,
        verbose=100,
    ) -> None:

        self.model = CatBoostRegressor(
            loss_function=loss_function,
            eval_metric=eval_metric,
            iterations=iterations,
            learning_rate=learning_rate,
            thread_count=thread_count,
            random_seed=random_seed,
            verbose=verbose,
        )

    def train(self, X_train, y_train, X_val, y_val, caracteristicas_categoricas):
        """Entrena y devuelve un modelo CatBoost."""

        print("Iniciando entrenamiento de CatBoost...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            cat_features=caracteristicas_categoricas,  # Manejo nativo
            early_stopping_rounds=100,
            use_best_model=True,
        )
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

        feature_importances = self.model.get_feature_importance()
        feature_names = self.model.feature_names_

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importances)
        plt.xlabel("Importancia de la característica")
        plt.title("Importancia de las características - CatBoost")

        if save_path:
            plt.savefig(save_path)
            print(f"Gráfico de importancia de características guardado en: {save_path}")
        else:
            plt.show()
