# run_training.py
from pathlib import Path
from demand_forecasting.walmart_data import WalmartDataloader, WalmartFeatures
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
    print("="*70)
    print("ENTRENAMIENTO DE MODELO DE FORECASTING - WALMART SALES")
    print("="*70)

    # Cargar datos
    print("\n1️⃣ CARGANDO DATOS...")
    data_loader = WalmartDataloader()
    df = data_loader.get_clean_data()

    # Usar LightGBM (modelo ganador)
    print("\n2️⃣ INICIALIZANDO MODELO: LightGBM")
    print("   Motivo: Mejor precisión (MAE $37,462) y feature importance")
    trainer = LightGBMTrainer()

    # Split temporal
    print(f"\n3️⃣ DIVIDIENDO DATOS (fecha corte: {FECHA_CORTE_VAL})...")
    print(f"   Features: {len(CARACTERISTICAS)}")
    X_train, y_train, X_val, y_val = split_data_by_date(
        df, FECHA_CORTE_VAL, CARACTERISTICAS, OBJETIVO
    )

    # Entrenar
    print("\n4️⃣ ENTRENANDO MODELO...")
    trainer.train(X_train, y_train, X_val, y_val, CARACTERISTICAS_CATEGORICAS)

    # Evaluar
    print("\n5️⃣ EVALUANDO MODELO...")
    wmae, mae = trainer.evaluate(X_val, y_val)

    # Calcular precisión
    media_ventas = df[OBJETIVO].mean()
    error_relativo = (mae / media_ventas) * 100
    precision = 100 - error_relativo

    print(f"\n{'='*70}")
    print(f"📊 MÉTRICAS FINALES")
    print(f"{'='*70}")
    print(f"   MAE:              ${mae:,.2f}")
    print(f"   WMAE:             ${wmae:,.2f}")
    print(f"   Error Relativo:   {error_relativo:.2f}%")
    print(f"   Precisión:        {precision:.2f}%")
    print(f"   Ventas Promedio:  ${media_ventas:,.2f}")
    print(f"{'='*70}")

    # Guardar modelo
    print(f"\n6️⃣ GUARDANDO MODELO...")
    model_path = MODELS_PATH / "lgbm_walmart_model.joblib"
    trainer.save(model_path)
    print(f"   ✓ Modelo guardado en: {model_path}")

    # Generar predicciones para plots
    print(f"\n7️⃣ GENERANDO VISUALIZACIONES...")
    predicciones = trainer.predict(X_val)
    val_data_original = df[
        df[WalmartFeatures.DATE.value] >= FECHA_CORTE_VAL
    ]

    # Plot predicciones vs reales
    plot_path = PLOTS_PATH / "predicciones_tienda_1.png"
    plot_predictions(
        val_data_original,
        predicciones,
        store_id=1,
        save_path=plot_path,
    )
    print(f"   ✓ Predicciones guardadas en: {plot_path}")

    # Plot feature importance
    importance_path = PLOTS_PATH / "importancia_caracteristicas.png"
    trainer.plot_feature_importance(save_path=importance_path)
    print(f"   ✓ Importancia features guardada en: {importance_path}")

    print(f"\n{'='*70}")
    print("✅ PIPELINE DE ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"\nArchivos generados:")
    print(f"   • Modelo:      {model_path}")
    print(f"   • Predicción:  {plot_path}")
    print(f"   • Importance:  {importance_path}")
    print()


if __name__ == "__main__":
    main()
