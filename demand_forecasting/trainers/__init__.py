from ..walmart_data import WalmartFeatures

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
