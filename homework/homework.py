import gzip
import json
import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def pregunta_01():
    # -------------------------------------------------------------------------
    # Paso 1. Limpieza de los datasets
    # -------------------------------------------------------------------------
    train_path = "files/input/train_data.csv.zip"
    test_path = "files/input/test_data.csv.zip"

    # [CORRECCIÓN]: Leemos correctamente usando la compresión nativa sin duplicados
    train_df = pd.read_csv(train_path, compression="zip")
    test_df = pd.read_csv(test_path, compression="zip")

    def clean_dataset(df):
        # Renombrar la columna objetivo
        if "default payment next month" in df.columns:
            df = df.rename(columns={"default payment next month": "default"})

        # Remover la columna ID
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])

        # Eliminar registros con información no disponible (nulos)
        df = df.dropna()

        # [MEJORA]: Agrupar valores > 4 o == 0 en EDUCATION en la categoría 4 (others)
        if "EDUCATION" in df.columns:
            df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
            df.loc[df["EDUCATION"] == 0, "EDUCATION"] = 4

        return df

    train_df = clean_dataset(train_df)
    test_df = clean_dataset(test_df)

    # -------------------------------------------------------------------------
    # Paso 2. División en x_train, y_train, x_test, y_test
    # -------------------------------------------------------------------------
    x_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]

    x_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    # -------------------------------------------------------------------------
    # Paso 3. Creación del Pipeline para el modelo
    # -------------------------------------------------------------------------
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                # Ajustamos los parámetros para un óptimo balance
                RandomForestClassifier(
                    random_state=42, 
                    n_jobs=-1, 
                    n_estimators=200, 
                    max_depth=None,
                    min_samples_split=2
                ),
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # Paso 4. Optimización de hiperparámetros usando validación cruzada
    # -------------------------------------------------------------------------
    param_grid = {
        "classifier__n_estimators": [200],
        "classifier__max_depth": [None],
        "classifier__min_samples_split": [2],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    # Entrenar optimizando con los datos de entrenamiento
    grid_search.fit(x_train, y_train)

    # -------------------------------------------------------------------------
    # Paso 5. Guardar el modelo comprimido con gzip
    # -------------------------------------------------------------------------
    models_dir = "files/models"
    os.makedirs(models_dir, exist_ok=True)
    model_output_path = os.path.join(models_dir, "model.pkl.gz")

    with gzip.open(model_output_path, "wb") as f:
        pickle.dump(grid_search, f)

    # -------------------------------------------------------------------------
    # Pasos 6 y 7. Calcular métricas, matrices de confusión y guardarlas en JSON
    # -------------------------------------------------------------------------
    output_dir = "files/output"
    os.makedirs(output_dir, exist_ok=True)
    metrics_output_path = os.path.join(output_dir, "metrics.json")

    y_train_pred = grid_search.predict(x_train)
    y_test_pred = grid_search.predict(x_test)

    all_metrics = []

    # 1. Métrica de entrenamiento
    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_train, y_train_pred)
        ),
        "recall": float(recall_score(y_train, y_train_pred)),
        "f1_score": float(f1_score(y_train, y_train_pred)),
    }
    all_metrics.append(metrics_train)

    # 2. Métrica de prueba
    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, y_test_pred)
        ),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1_score": float(f1_score(y_test, y_test_pred)),
    }
    all_metrics.append(metrics_test)

    # 3. Matriz de confusión de entrenamiento
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_train_dict = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0, 0]),
            "predicted_1": int(cm_train[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_train[1, 0]),
            "predicted_1": int(cm_train[1, 1]),
        },
    }
    all_metrics.append(cm_train_dict)

    # 4. Matriz de confusión de prueba
    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_test_dict = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0, 0]),
            "predicted_1": int(cm_test[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_test[1, 0]),
            "predicted_1": int(cm_test[1, 1]),
        },
    }
    all_metrics.append(cm_test_dict)

    # Guardar en el archivo metrics.json línea por línea (formato JSON Lines)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        for metric in all_metrics:
            f.write(json.dumps(metric) + "\n")


if __name__ == "__main__":
    pregunta_01()