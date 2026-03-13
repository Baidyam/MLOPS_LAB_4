# Airflow Wine Clustering Lab

This project implements an **automated ML pipeline using Apache Airflow**, designed as a lab exercise for workflow orchestration.  
It is based on an original credit-card clustering lab but includes **several key modifications** to make it unique.

---

## Key Changes from Original Repo

1. **Dataset Change**
   - Original lab used a credit card dataset (`file.csv`).
   - This lab uses the **Wine dataset** from `sklearn.datasets`.
   - Features selected for clustering:
     ```
     alcohol, malic_acid, ash, proline
     ```

2. **Algorithm Change**
   - Original lab used **K-Means clustering**.
   - This lab uses **Agglomerative Clustering** (Hierarchical clustering), implemented via `sklearn.cluster.AgglomerativeClustering`.
   - This demonstrates flexibility in clustering approaches.

3. **Airflow Improvements**
   - Added **DAG scheduling**: the pipeline now runs **daily** (`schedule='@daily'`).
   - Added **task retries**: each task retries up to 2 times on failure with a 2-minute delay.
   - Added **Airflow tags**: `['ml','clustering','wine']`.
   - Added **logging** within tasks for better traceability in Airflow logs.

4. **Scaling Method**
   - Original lab used `MinMaxScaler`.
   - This lab uses `StandardScaler` for feature normalization.

---

## Pipeline Overview

The DAG is named: `Wine_Clustering_Airflow`

The pipeline steps:

1. **Load Data**  
   Load the Wine dataset directly from `sklearn.datasets`.

2. **Data Preprocessing**  
   Select relevant features and scale them using `StandardScaler`.

3. **Build & Save Model**  
   Train an **Agglomerative Clustering** model and save it to the `model` directory.

4. **Load Model & Predict**  
   Load the saved model and predict the cluster for a new test sample (`test.csv`).

---


---

## How to Run

1. Setup Airflow environment:

```bash
bash setup.sh
```

2. Start Airflow containers:
   docker compose up -d
 


