# demand_forecasting/model.py
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
from abc import ABC, abstractmethod

def train_loggin(func):
    def wrapper(*args, **kwargs):
        print("Iniciando entrenamiento...")
        res = func(*args, **kwargs)
        print("¡Entrenamiento completado!")
        return res
    return wrapper

def evaluate_loggin(func):
    def wrapper(*args, **kwargs):
        print("Iniciando evaluación del modelo...")
        res = func(*args, **kwargs)
        print("Evaluación completada.")
        return res
    return wrapper


class BaseModel(ABC):
    @abstractmethod
    @train_loggin
    def train(self, X_train, y_train, X_val, y_val, caracteristicas_categoricas):
        """Entrena el modelo con los datos proporcionados."""
        pass

    @abstractmethod
    @evaluate_loggin
    def evaluate(self, X_val, y_val):
        """Evalúa el modelo con los datos de validación y devuelve las métricas."""
        pass

    @abstractmethod
    def predict(self, X):
        """Genera predicciones utilizando el modelo entrenado."""
        pass
    
    @abstractmethod
    def save(self, filepath: Path):
        """Guarda el modelo entrenado en un archivo."""
        pass

# Decoradores para el logging

