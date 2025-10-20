from ..walmart_data import WalmartFeatures

# Import all the class from the trainers
from .XGBoost import XGBoostTrainer
from .lightGBM import LightGBMTrainer
from .catboost import CatBoostTrainer
from .randomforest import RandomForestTrainer
from .prophet import ProphetTrainer
from .neural_network import LSTMTrainer

def split_data_by_date(df, fecha_corte_val, caracteristicas, objetivo):
    """Divide el dataframe en train/validation por fecha."""

    # Set de Entrenamiento
    train_df = df[df[WalmartFeatures.DATE] < fecha_corte_val].set_index(WalmartFeatures.DATE)
    X_train = train_df[caracteristicas]
    y_train = train_df[objetivo]

    # Set de Validación
    val_df = df[df[WalmartFeatures.DATE] >= fecha_corte_val].set_index(WalmartFeatures.DATE)
    X_val = val_df[caracteristicas]
    y_val = val_df[objetivo]

    print(f"Datos de Entrenamiento: {X_train.shape[0]} filas")
    print(f"Datos de Validación:   {X_val.shape[0]} filas")

    return X_train, y_train, X_val, y_val
