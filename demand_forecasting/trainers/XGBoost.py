# demand_forecasting/xgboost_model.py
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
from typing import Optional
from .base import BaseModel
from ..walmart_data import WalmartFeatures


class XGBoostTrainer(BaseModel):
    def __init__(
        self,
        objective="reg:squarederror",  # Objetivo común para regresión en XGB
        eval_metric="mae",
        n_estimators=1000,
        learning_rate=0.01,
        n_jobs=-1,
        random_state=42,
        enable_categorical=True,  # Para manejo moderno de categóricas
    ) -> None:

        self.model = xgb.XGBRegressor(
            objective=objective,
            eval_metric=eval_metric,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
            enable_categorical=enable_categorical,  # Requiere que las dtypes sean 'category'
        )

    def train(self, X_train, y_train, X_val, y_val, caracteristicas_categoricas):
        """Entrena y devuelve un modelo XGBoost."""

        print("Iniciando entrenamiento de XGBoost...")

        # Para que 'enable_categorical=True' funcione, las columnas
        # categóricas en X_train y X_val deben ser de tipo 'category'
        for col in caracteristicas_categoricas:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")

        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        print("¡Entrenamiento completado!")
        return self.model

    def evaluate(self, X_val, y_val):
        """Calcula y muestra las métricas de error WMAE y MAE."""

        predicciones_val = self.model.predict(X_val)

        # Calcular WMAE (Métrica clave de Walmart)
        weights = (
            X_val[WalmartFeatures.HOLIDAY_FLAG.value]
            .apply(lambda x: 5 if x else 1)
            .astype(float)
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

    def plot_feature_importance(self, save_path: Optional[Path] = None):
        """Grafica la importancia de las características del modelo."""
        import matplotlib.pyplot as plt

        xgb.plot_importance(self.model)
        if save_path:
            plt.savefig(save_path)
            print(f"Importancia de características guardada en: {save_path}")
        plt.show()
