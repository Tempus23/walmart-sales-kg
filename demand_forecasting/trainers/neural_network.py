
# demand_forecasting/trainers/neural_network.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from .base import BaseModel
from ..walmart_data import WalmartFeatures


class LSTMTrainer(BaseModel):
    """
    Entrenador para un modelo LSTM (Long Short-Term Memory).
    Este modelo requiere que los datos de entrada estén escalados.
    """

    def __init__(self, timesteps=1, epochs=50, batch_size=32, validation_split=0.2):
        self.timesteps = timesteps
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def _prepare_data(self, X, y=None):
        """Escala y remodela los datos para el LSTM."""
        
        # Escalar características
        X_scaled = self.scaler_X.transform(X)
        
        # Remodelar para LSTM: [samples, timesteps, features]
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], self.timesteps, X_scaled.shape[1]))
        
        if y is not None:
            # Escalar objetivo
            y_scaled = self.scaler_y.transform(y.values.reshape(-1, 1))
            return X_reshaped, y_scaled
            
        return X_reshaped

    def train(self, X_train, y_train, X_val, y_val, caracteristicas_categoricas):
        """Entrena un modelo LSTM."""
        
        print("Iniciando entrenamiento de LSTM...")

        # Convertir nombres de columnas a string para evitar errores de sklearn
        X_train.columns = [str(col) for col in X_train.columns]
        X_val.columns = [str(col) for col in X_val.columns]

        # 1. Ajustar los escaladores SOLO con datos de entrenamiento
        self.scaler_X.fit(X_train)
        self.scaler_y.fit(y_train.values.reshape(-1, 1))

        # 2. Preparar datos de entrenamiento y validación
        X_train_prepared, y_train_prepared = self._prepare_data(X_train, y_train)
        X_val_prepared, y_val_prepared = self._prepare_data(X_val, y_val)

        # 3. Construir el modelo
        n_features = X_train.shape[1]
        self.model = Sequential([
            Input(shape=(self.timesteps, n_features)),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        print(self.model.summary())

        # 4. Entrenar el modelo
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        self.model.fit(
            X_train_prepared,
            y_train_prepared,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val_prepared, y_val_prepared),
            callbacks=[early_stopping],
            verbose=1,
        )
        
        print("¡Entrenamiento completado!")
        return self.model

    def predict(self, X):
        """Genera predicciones y las devuelve a la escala original."""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a .train() primero.")
            
        # Asegurarse de que las columnas son strings
        X.columns = [str(col) for col in X.columns]

        # Preparar datos
        X_prepared = self._prepare_data(X)
        
        # Predecir
        predictions_scaled = self.model.predict(X_prepared)
        
        # Invertir escalado para obtener valor real
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions.flatten()

    def evaluate(self, X_val, y_val):
        """Calcula las métricas WMAE y MAE."""
        
        print("Evaluando modelo LSTM...")
        predicciones_val = self.predict(X_val)
        
        # Calcular WMAE
        weights = X_val[WalmartFeatures.HOLIDAY_FLAG.value].apply(lambda x: 5 if x else 1)
        error_absoluto = np.abs(y_val - predicciones_val)
        wmae = (np.sum(weights * error_absoluto)) / (np.sum(weights))
        
        mae_normal = mean_absolute_error(y_val, predicciones_val)

        print(f"======================================================")
        print(f"Error Absoluto Medio (MAE) Normal:    ${mae_normal:,.2f}")
        print(f"Error Absoluto Medio PONDERADO (WMAE): ${wmae:,.2f}")
        print(f"======================================================")
        
        return wmae, mae_normal

    def save(self, filepath: Path):
        """Guarda el modelo Keras y los escaladores."""
        model_path = filepath.with_suffix('.h5')
        scalers_path = filepath.with_suffix('.pkl')
        
        print(f"Guardando modelo Keras en: {model_path}")
        self.model.save(model_path)
        
        print(f"Guardando escaladores en: {scalers_path}")
        joblib.dump({'scaler_X': self.scaler_X, 'scaler_y': self.scaler_y}, scalers_path)

    def plot_feature_importance(self, save_path: Path = None):
        """La importancia de características no es directamente trivial en LSTMs."""
        print("La gráfica de importancia de características no está implementada para el modelo LSTM.")
        pass
