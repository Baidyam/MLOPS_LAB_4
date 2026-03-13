import pandas as pd
import logging
import pickle
import base64
import os

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


def load_data():

    logging.info("Loading Wine dataset")

    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)

    serialized = pickle.dumps(df)

    return base64.b64encode(serialized).decode("ascii")


def data_preprocessing(data_b64: str):

    logging.info("Preprocessing dataset")

    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()

    clustering_data = df[
        [
            "alcohol",
            "malic_acid",
            "ash",
            "proline"
        ]
    ]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(clustering_data)

    serialized = pickle.dumps(scaled_data)

    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, filename: str):

    logging.info("Building Agglomerative Clustering model")

    data_bytes = base64.b64decode(data_b64)
    data = pickle.loads(data_bytes)

    model = AgglomerativeClustering(n_clusters=3)

    labels = model.fit_predict(data)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "model"
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    logging.info("Model saved successfully")

    return labels.tolist()


def load_model_elbow(filename: str, labels: list):

    logging.info("Loading saved model")

    model_path = os.path.join(
        os.path.dirname(__file__),
        "../model",
        filename
    )

    model = pickle.load(open(model_path, "rb"))

    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../data/test.csv")
    )

    prediction = model.fit_predict(df)

    logging.info(f"Prediction cluster: {prediction[0]}")

    return int(prediction[0])
