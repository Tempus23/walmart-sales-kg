# demand_forecasting/prophet_model.py
from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
from .base import BaseModel
from ..walmart_data import WalmartFeatures
import warnings

# Suprimir los logs de 'cmdstanpy' que son muy ruidosos
warnings.filterwarnings('ignore', category=FutureWarning, module='prophet')
warnings.filterwarnings('ignore', category=UserWarning, module='prophet')

class ProphetTrainer(BaseModel):
    """
    Entrenador para Prophet. Este modelo es fundamentalmente diferente.
    
    Es un modelo univariante, por lo que entrenaremos UN MODELO SEPARADO
    por cada tienda ('Store').
    
    Esta clase actúa como un "contenedor" que gestiona la colección de modelos.
    """
    def __init__(self, **prophet_kwargs) -> None:
        """
        Inicializa el contenedor de modelos.
        'prophet_kwargs' se pasarán a cada instancia de Prophet
        (ej. growth='logistic', seasonality_mode='multiplicative')
        """
        self.models = {} # Un diccionario para guardar {store_id: fitted_model}
        self.prophet_kwargs = prophet_kwargs
        
    def train(self, X_train, y_train, X_val, y_val, caracteristicas_categoricas):
        """
        Entrena un modelo Prophet separado para CADA tienda.
        """
        
        print("Iniciando entrenamiento de Prophet (un modelo por tienda)...")
        
        # 1. Reconstruir el DataFrame de entrenamiento
        # Prophet necesita el índice (fecha) como una columna 'ds'
        # y el objetivo ('y')
        df_train = X_train.copy()
        df_train[WalmartFeatures.WEEKLY_SALES.value] = y_train
        df_train = df_train.reset_index().rename(columns={'Date': 'ds'})
        
        # 2. Identificar las tiendas únicas
        stores = df_train[WalmartFeatures.STORE.value].unique()
        
        # 3. Identificar regresores (features)
        # Prophet los llama "regresores adicionales".
        # Excluimos la tienda (ya que filtramos por ella) y el objetivo
        regressors = [
            col for col in X_train.columns 
            if col not in [WalmartFeatures.STORE.value]
        ]
        
        for store_id in stores:
            print(f"  Entrenando modelo para Tienda {store_id}...")
            
            # 4. Preparar datos para esta tienda
            store_df = df_train[
                df_train[WalmartFeatures.STORE.value] == store_id
            ].rename(columns={WalmartFeatures.WEEKLY_SALES.value: 'y'})
            # Asegurarse de que 'ds' está presente
            if 'ds' not in store_df.columns:
                store_df = store_df.reset_index().rename(columns={WalmartFeatures.DATE.value: 'ds'})
                store_df['y'] = store_df[WalmartFeatures.WEEKLY_SALES.value]

            # 5. Inicializar Prophet
            # Podemos añadir festivos automáticamente, pero es mejor
            # usar 'IsHoliday' como un regresor.
            model = Prophet(**self.prophet_kwargs)
            
            # 6. Añadir todos los 'features' como regresores
            for regressor in regressors:
                model.add_regressor(regressor)
            
            # 7. Entrenar el modelo
            model.fit(store_df[['ds', 'y'] + regressors])
            
            # 8. Guardar el modelo entrenado en nuestro diccionario
            self.models[store_id] = model
            
        print(f"\n¡Entrenamiento completado! {len(self.models)} modelos Prophet entrenados.")
        return self.models
    
    def predict(self, X):
        """
        Genera predicciones iterando sobre cada modelo de tienda.
        """
        if not self.models:
            raise ValueError("No hay modelos entrenados. Llama a .train() primero.")
            
        all_predictions = []
        
        # Preparamos el dataframe de 'futuro' (que es X)
        X_future = X.reset_index().rename(columns={'Date': 'ds'})
        stores = X_future[WalmartFeatures.STORE.value].unique()
        
        for store_id in stores:
            if store_id not in self.models:
                print(f"Advertencia: No hay modelo entrenado para la tienda {store_id}. Saltando.")
                continue
                
            # 1. Obtener el modelo y los datos para esta tienda
            model = self.models[store_id]
            future_df_store = X_future[
                X_future[WalmartFeatures.STORE.value] == store_id
            ]
            
            # 2. Realizar la predicción
            # Prophet predice sobre el dataframe 'future'
            forecast = model.predict(future_df_store)
            
            # 3. Guardar solo la predicción ('yhat')
            # Importante: alinear con el índice original de X
            preds = pd.Series(forecast['yhat'].values, index=future_df_store.index)
            all_predictions.append(preds)
            
        if not all_predictions:
            raise ValueError("No se pudieron generar predicciones.")
            
        # Concatenar todas las predicciones de series
        final_preds = pd.concat(all_predictions)
        
        # Re-ordenar al orden original del DataFrame X
        final_preds = final_preds.sort_index()
        final_preds.index = X.index
        return final_preds

    def evaluate(self, X_val, y_val):
        """
        Calcula las métricas WMAE y MAE usando el método .predict()
        """
        
        print("Evaluando modelos Prophet...")
        predicciones_val = self.predict(X_val)
        
        # Calcular WMAE (Métrica clave de Walmart)
        weights = X_val[WalmartFeatures.HOLIDAY_FLAG.value].apply(lambda x: 5 if x else 1)
        error_absoluto = np.abs(y_val - predicciones_val)
        wmae = (np.sum(weights * error_absoluto)) / (np.sum(weights))
        
        mae_normal = mean_absolute_error(np.asarray(y_val), np.asarray(predicciones_val))

        print(f"======================================================")
        print(f"Error Absoluto Medio (MAE) Normal:    ${mae_normal:,.2f}")
        print(f"Error Absoluto Medio PONDERADO (WMAE): ${wmae:,.2f}")
        print(f"======================================================")
        
        return wmae, mae_normal
    
    def save(self, filepath: Path):
        """
Aviso: Guardar un diccionario de modelos Prophet puede ser un
archivo MUY grande.
"""
        print(f"Guardando diccionario de modelos Prophet en: {filepath}")
        joblib.dump(self.models, filepath)
        
    @classmethod
    def load(cls, filepath: Path):
        """
        Carga un diccionario de modelos Prophet guardado.
        """
        print(f"Cargando modelos desde: {filepath}")
        models_dict = joblib.load(filepath)
        
        # Crea una nueva instancia e inyecta los modelos cargados
        instance = cls()
        instance.models = models_dict
        return instance
    
    def plot_feature_importance(self, save_path: Path = None):
        pass