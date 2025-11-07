# run_training.py
from pathlib import Path
from demand_forecasting.walmart_data import WalmartDataloader, WalmartFeatures
from demand_forecasting.model import train_model, evaluate_model, save_model
from demand_forecasting.plotting import plot_predictions

from demand_forecasting.trainers import (
    split_data_by_date,
    LightGBMTrainer,
    XGBoostTrainer,
    CatBoostTrainer,
    RandomForestTrainer,
    ProphetTrainer,
    LSTMTrainer
)

# --- 0. Configuración del Proyecto ---
OUTPUT_PATH = Path("outputs")
MODELS_PATH = OUTPUT_PATH / "models"
PLOTS_PATH = OUTPUT_PATH / "plots"

# Crear directorios de salida
MODELS_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Constantes del modelo
FECHA_CORTE_VAL = "2012-05-01"
OBJETIVO = WalmartFeatures.WEEKLY_SALES.value
CARACTERISTICAS = [
    WalmartFeatures.STORE.value,
    WalmartFeatures.HOLIDAY_FLAG.value,
    WalmartFeatures.MONTH.value,
    WalmartFeatures.QUARTER.value,
    WalmartFeatures.YEAR.value,
    WalmartFeatures.WEEK_OF_YEAR.value,
    WalmartFeatures.MONTH_SIN.value,
    WalmartFeatures.MONTH_COS.value,
    WalmartFeatures.WEEK_SIN.value,
    WalmartFeatures.WEEK_COS.value,
    WalmartFeatures.VENTAS_LAG_1.value,
    WalmartFeatures.VENTAS_LAG_4.value,
    WalmartFeatures.VENTAS_LAG_52.value,
    WalmartFeatures.MEDIA_MOVIL_4_SEMANAS.value,
    WalmartFeatures.MEDIA_MOVIL_12_SEMANAS.value,
]
CARACTERISTICAS_CATEGORICAS = [
    WalmartFeatures.STORE.value,
    WalmartFeatures.HOLIDAY_FLAG.value,
    WalmartFeatures.MONTH.value,
    WalmartFeatures.QUARTER.value,
    WalmartFeatures.YEAR.value,
    WalmartFeatures.WEEK_OF_YEAR.value,
]


def main():
    data_loader = WalmartDataloader()
    df = data_loader.get_clean_data()
    trainer = LSTMTrainer()

    X_train, y_train, X_val, y_val = split_data_by_date(
        df, FECHA_CORTE_VAL, CARACTERISTICAS, OBJETIVO
    )

    trainer.train(X_train, y_train, X_val, y_val, CARACTERISTICAS_CATEGORICAS)

    trainer.evaluate(X_val, y_val)

    trainer.save(MODELS_PATH / "lgbm_walmart_model.joblib")

    predicciones = trainer.predict(X_val)
    val_data_original = df[
        df[WalmartFeatures.DATE.value] >= FECHA_CORTE_VAL
    ]  # Necesitamos el df original

    plot_predictions(
        val_data_original,
        predicciones,
        store_id=1,
        save_path=PLOTS_PATH / "predicciones_tienda_1.png",
    )

    trainer.plot_feature_importance(
        save_path=PLOTS_PATH / "importancia_caracteristicas.png"
    )

    print("\n¡Pipeline de entrenamiento completado!")


if __name__ == "__main__":
    main()
