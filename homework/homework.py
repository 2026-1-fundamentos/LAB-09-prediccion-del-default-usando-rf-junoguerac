# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

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