# demand_forecasting/plotting.py
import matplotlib.pyplot as plt
import lightgbm as lgb

def plot_predictions(val_data_original, predicciones, store_id=1, save_path=None):
    """Grafica las predicciones vs. los valores reales para una tienda."""
    
    resultados_val = val_data_original.copy()
    resultados_val['Prediccion'] = predicciones
    
    df_plot = resultados_val[resultados_val['Store'] == store_id]

    plt.figure(figsize=(20, 7))
    plt.plot(df_plot.index, df_plot['Weekly_Sales'], label='Ventas Reales', color='blue', alpha=0.7)
    plt.plot(df_plot.index, df_plot['Prediccion'], label='Predicción del Modelo', color='red', linestyle='--')
    plt.title(f'Comparativa Real vs. Predicción (Tienda {store_id})')
    plt.legend()
    
    if save_path:
        print(f"Guardando gráfico de predicciones en: {save_path}")
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(model, save_path=None):
    """Grafica la importancia de las características del modelo."""
    
    lgb.plot_importance(model, figsize=(10, 8), max_num_features=15, height=0.7)
    plt.title('Importancia de las Características (Feature Importance)')
    plt.tight_layout()
    
    if save_path:
        print(f"Guardando gráfico de importancia en: {save_path}")
        plt.savefig(save_path)
    plt.show()