# demand_forecasting/model.py
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
from .walmart_data import WalmartFeatures

def split_data_by_date(df, fecha_corte_val, caracteristicas, objetivo):
    """Divide el dataframe en train/validation por fecha."""
    
    # Set de Entrenamiento
    train_data = df[df[WalmartFeatures.DATE] < fecha_corte_val]
    X_train = train_data[caracteristicas]
    y_train = train_data[objetivo]

    # Set de Validación
    val_data = df[df[WalmartFeatures.DATE] >= fecha_corte_val]
    X_val = val_data[caracteristicas]
    y_val = val_data[objetivo]
    
    print(f"Datos de Entrenamiento: {X_train.shape[0]} filas")
    print(f"Datos de Validación:   {X_val.shape[0]} filas")
    
    return X_train, y_train, X_val, y_val

def train_model(X_train, y_train, X_val, y_val, caracteristicas_categoricas):
    """Entrena y devuelve un modelo LightGBM."""
    
    model = lgb.LGBMRegressor(
        objective='regression_l1',
        metric='l1',
        n_estimators=1000,
        learning_rate=0.01,
        n_jobs=-1,
        random_state=42
    )
    
    print("Iniciando entrenamiento...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l1',
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(period=100)],
        #categorical_feature=caracteristicas_categoricas
    )
    print("¡Entrenamiento completado!")
    return model

def evaluate_model(model, X_val, y_val):
    """Calcula y muestra las métricas de error WMAE y MAE."""
    
    predicciones_val = model.predict(X_val)
    
    # Calcular WMAE (Métrica clave de Walmart)
    weights = X_val[WalmartFeatures.HOLIDAY_FLAG.value].apply(lambda x: 5 if x else 1)
    error_absoluto = np.abs(y_val - predicciones_val)
    wmae = (np.sum(weights * error_absoluto)) / (np.sum(weights))
    
    mae_normal = mean_absolute_error(y_val, predicciones_val)

    print(f"======================================================")
    print(f"Error Absoluto Medio (MAE) Normal:    ${mae_normal:,.2f}")
    print(f"Error Absoluto Medio PONDERADO (WMAE): ${wmae:,.2f}")
    print(f"======================================================")
    
    return wmae, mae_normal

def save_model(model, filepath: Path):
    """Guarda el modelo entrenado en un archivo."""
    print(f"Guardando modelo en: {filepath}")
    joblib.dump(model, filepath)