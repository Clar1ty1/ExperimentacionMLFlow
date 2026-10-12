import argparse
import json
import pandas
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo California Housing"
    )

    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def load_data():
    return pandas.read_csv("housing.csv")

def clean_data(df: pandas.DataFrame) -> pandas.DataFrame:
    median_bedrooms = df["total_bedrooms"].median()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(median_bedrooms)
    return df

def split_train(df: pandas.DataFrame): 
    # Separamos variable objetivo y predictoras 
    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1765,
        random_state=42
    )
    return X_test, y_test, X_train, y_train, X_val, y_val

def get_preprocessor():
    cat_cols = ["ocean_proximity"]

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    return ColumnTransformer([
        ("cat", categorical_transformer, cat_cols)
    ])

# Se cambio de linear regression a random forest con el fin de poder utilizar mas parametros desde la terminal
def baseline(preprocessor, X_train, y_train, args):
    model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1
    ))])

    model.fit(X_train, y_train)

    return model

def eval(model, X_val, y_val, X_test, y_test):
    results = {}

    for name, X_part, y_part in [
        ("val", X_val, y_val),
        ("test", X_test, y_test)
    ]:
        pred = model.predict(X_part)

        results[f"{name}_MAE"] = mean_absolute_error(y_part, pred)
        results[f"{name}_RMSE"] = np.sqrt(
            mean_squared_error(y_part, pred)
        )
        results[f"{name}_R2"] = r2_score(y_part, pred)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return results


if __name__ == "__main__":
    args = parse_args()

    df = load_data()

    df = clean_data(df)

    X_test, y_test, X_train, y_train, X_val, y_val = split_train(df)

    preprocessor = get_preprocessor()

    mlflow.set_experiment("California Housing - Random Forest")

    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("seed", args.seed)

        model = baseline(preprocessor, X_train, y_train, args)

        results = eval(model, X_val, y_val, X_test, y_test)

        # Registrar métricas de validación y test
        for metric, value in results.items():
            mlflow.log_metric(metric, value)

        # Guardar las métricas como artefacto
        with open("metrics.json", "w") as f:
            json.dump(results, f, indent=2)

        mlflow.log_artifact("metrics.json")

        # Guardar el modelo entrenado se utiliza pickle para resolver el problema de confianza de skops (mlflow considera no confiable un tipo interno de scikit learn)
        mlflow.sklearn.log_model(model, name="model", serialization_format="pickle")