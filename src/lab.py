import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from kneed import KneeLocator
import pickle
import os
import base64


def load_data():
    """
    Load wine dataset from sklearn.
    """

    print("Loading Wine dataset")

    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)

    serialized_data = pickle.dumps(df)

    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):

    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()

    clustering_data = df[[
        "alcohol",
        "malic_acid",
        "ash",
        "proline"
    ]]

    scaler = MinMaxScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    serialized = pickle.dumps(clustering_data_scaled)

    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, filename: str):

    data_bytes = base64.b64decode(data_b64)
    data = pickle.loads(data_bytes)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42
    }

    sse = []

    for k in range(1, 20):

        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)

        sse.append(kmeans.inertia_)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "model"
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)

    return sse


def load_model_elbow(filename: str, sse: list):

    model_path = os.path.join(
        os.path.dirname(__file__),
        "../model",
        filename
    )

    loaded_model = pickle.load(open(model_path, "rb"))

    kl = KneeLocator(range(1, 20), sse, curve="convex", direction="decreasing")

    print(f"Optimal number of clusters: {kl.elbow}")

    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../data/test.csv")
    )

    prediction = loaded_model.predict(df)[0]

    return int(prediction)
