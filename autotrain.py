# autotrain.py
from pathlib import Path
from demand_forecasting.walmart_data import WalmartDataloader, WalmartFeatures
from demand_forecasting.trainers import split_data_by_date
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

# --- 0. Configuración del Proyecto ---
OUTPUT_PATH = Path("outputs")
PIPELINE_PATH = OUTPUT_PATH / "automl_pipelines"

# Crear directorios de salida
PIPELINE_PATH.mkdir(parents=True, exist_ok=True)

# Constantes del modelo
FECHA_CORTE_VAL = "2012-05-01"
OBJETIVO = WalmartFeatures.WEEKLY_SALES.value


def main():
    # --- 1. Carga y Preparación de Datos ---
    print("Cargando y preparando datos...")
    data_loader = WalmartDataloader()
    df = data_loader.get_clean_data()

    # Usar todas las características excepto el objetivo y la fecha
    caracteristicas = [
        col.value for col in WalmartFeatures if col.value not in [OBJETIVO, WalmartFeatures.DATE.value]
    ]

    X_train, y_train, X_val, y_val = split_data_by_date(
        df, FECHA_CORTE_VAL, caracteristicas, OBJETIVO
    )

    # --- 2. Configuración de TPOT ---
    # TPOT puede tardar mucho tiempo. 
    # Para una prueba inicial, usamos pocas generaciones y población.
    # Para una búsqueda real, aumenta `generations` y `population_size`.
    tpot_config = {
        'generations': 5, 
        'population_size': 20,

    }

    tpot = TPOTRegressor(**tpot_config)

    # --- 3. Ejecución de la Búsqueda ---
    print("\nIniciando búsqueda de pipeline con TPOT...")
    print("Esto puede tardar un tiempo considerable.")
    
    tpot.fit(X_train, y_train)

    # --- 4. Resultados ---
    print(f"\n¡Búsqueda completada!")
    print(f"Mejor puntuación (MAE): {-tpot.score(X_val, y_val):.2f}")

    # --- 5. Exportar el Mejor Pipeline ---
    pipeline_file = PIPELINE_PATH / 'tpot_best_pipeline.py'
    print(f"Exportando el mejor pipeline a: {pipeline_file}")
    tpot.export(str(pipeline_file))

    print("\nPuedes inspeccionar el archivo .py generado para ver el código del mejor modelo encontrado.")


if __name__ == "__main__":
    main()
