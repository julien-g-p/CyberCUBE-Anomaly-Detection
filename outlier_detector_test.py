# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import os

# Set the file name inside the Kaggle dataset, including extension
file_path = "hybrid_extended_20250819_151521.csv"
if not file_path:
    raise ValueError("Set file_path to the Kaggle dataset file name, including extension.")

# Load the latest version
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "kaushlenduparasar/satellite-communication-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documentation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())


def get_feature_matrix(dataframe):
    numeric_df = dataframe.select_dtypes(include=[np.number])
    return numeric_df.values


def create_isolation_forest(random_state=42, contamination=0.1, n_estimators=100):
    return IsolationForest(random_state=random_state, contamination=contamination, n_estimators=n_estimators)


def compute_scores(model, X):
    anomaly_scores = model.decision_function(X)
    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    if max_score == min_score:
        trust_scores = np.ones_like(anomaly_scores)
    else:
        trust_scores = (anomaly_scores - min_score) / (max_score - min_score)
    return anomaly_scores, trust_scores


def compute_average_scores(df_results):
    return df_results["anomaly_score"].mean(), df_results["trust_score"].mean()


if __name__ == "__main__":
    X_train = get_feature_matrix(df)

    n_estimators = 100
    contamination = 0.1
    model = create_isolation_forest(n_estimators=n_estimators, contamination=contamination)
    model.fit(X_train)

    anomaly_scores, trust_scores = compute_scores(model, X_train)
    df_results = df.copy()
    df_results["anomaly_score"] = anomaly_scores
    df_results["trust_score"] = trust_scores
    df_results["model_type"] = "IsolationForest"
    df_results["n_estimators"] = n_estimators

    most_anomalous = df_results.sort_values(
        by="anomaly_score"
    ).head(20)

    mean_anomaly, mean_trust = compute_average_scores(df_results)
    anomaly_by_event = df_results.groupby("event_type")["anomaly_score"].mean()
    trust_by_event = df_results.groupby("event_type")["trust_score"].mean()

    print("First 5 scored records:\n", df_results[["anomaly_score", "trust_score"]].head())
    print("\nTop 20 most anomalous records:\n", most_anomalous[
        [
            "event_type",
            "wired_latency_ms",
            "satellite_latency_ms",
            "wired_packet_loss_pct",
            "satellite_packet_loss_pct",
            "anomaly_score"
        ]
    ])
    print("Predictions:", model.predict(X_train))
    print(f"\nAverage anomaly score: {mean_anomaly:.6f}")
    print(f"Average trust score: {mean_trust:.6f}")
    print("\nAverage anomaly score by event_type:\n", anomaly_by_event)
    print("\nAverage trust score by event_type:\n", trust_by_event)

    # ============================================================================
    # RESEARCH-ORIENTED SUMMARY FILES
    # ============================================================================
    # These lightweight CSV files are designed for anomaly-detection research:
    # - Small enough for GitHub version control
    # - Capture key model performance metrics across configurations
    # - Enable comparative analysis of different n_estimators and contamination values
    # - Suitable for inclusion in academic publications and research repositories
    # ============================================================================

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # FILE 1: summary_metrics.csv
    # ============================================================================
    # PURPOSE: Captures overall model performance and anomaly distribution.
    # RESEARCH VALUE: Enables quick comparison of different hyperparameter
    # configurations and provides macro-level insights into the dataset's
    # anomalous characteristics.
    # ============================================================================
    summary_metrics = pd.DataFrame({
        "model_type": ["IsolationForest"],
        "n_estimators": [n_estimators],
        "contamination": [contamination],
        "avg_anomaly_score": [mean_anomaly],
        "avg_trust_score": [mean_trust],
        "normal_trust_score": [df_results[df_results["event_type"] == "normal"]["trust_score"].mean()],
        "wired_congestion_trust_score": [df_results[df_results["event_type"] == "wired_congestion"]["trust_score"].mean()],
        "software_bug_satellite_trust_score": [df_results[df_results["event_type"] == "software_bug_satellite"]["trust_score"].mean()],
        "satellite_handoff_trust_score": [df_results[df_results["event_type"] == "satellite_handoff"]["trust_score"].mean()],
        "wired_config_failure_trust_score": [df_results[df_results["event_type"] == "wired_config_failure"]["trust_score"].mean()],
    })
    
    # Calculate trust gap: the difference between normal and most critical failure mode
    normal_ts = summary_metrics["normal_trust_score"].values[0]
    wired_config_ts = summary_metrics["wired_config_failure_trust_score"].values[0]
    summary_metrics["trust_gap"] = normal_ts - wired_config_ts
    
    summary_path = os.path.join(results_dir, f"summary_metrics_IsolationForest_{n_estimators}.csv")
    summary_metrics.to_csv(summary_path, index=False)
    print(f"\nSaved summary metrics to: {summary_path}")

    # FILE 2: event_type_statistics.csv
    # ============================================================================
    # PURPOSE: Provides per-anomaly-type statistics (count, anomaly score ranges,
    # trust score statistics) for each event type in the dataset.
    # RESEARCH VALUE: Reveals which event types are most anomalous and where the
    # model has highest/lowest confidence. Useful for understanding model bias
    # and event-specific sensitivity.
    # ============================================================================
    event_stats = []
    for event_type in df_results["event_type"].unique():
        event_data = df_results[df_results["event_type"] == event_type]
        event_stats.append({
            "model_type": "IsolationForest",
            "n_estimators": n_estimators,
            "event_type": event_type,
            "count": len(event_data),
            "avg_anomaly_score": event_data["anomaly_score"].mean(),
            "avg_trust_score": event_data["trust_score"].mean(),
            "min_trust_score": event_data["trust_score"].min(),
            "max_trust_score": event_data["trust_score"].max(),
        })
    
    event_type_stats_df = pd.DataFrame(event_stats)
    event_stats_path = os.path.join(results_dir, f"event_type_statistics_IsolationForest_{n_estimators}.csv")
    event_type_stats_df.to_csv(event_stats_path, index=False)
    print(f"Saved event type statistics to: {event_stats_path}")

    # FILE 3: top_20_anomalies.csv
    # ============================================================================
    # PURPOSE: Extracts the 20 most anomalous records with full diagnostic data.
    # RESEARCH VALUE: Enables deep investigation of model-detected anomalies,
    # helping validate model behavior and identify dataset characteristics.
    # Sorted by anomaly_score (ascending) to show strongest anomalies first.
    # ============================================================================
    top_20_columns = [
        "timestamp",
        "terminal_id",
        "event_type",
        "wired_latency_ms",
        "satellite_latency_ms",
        "wired_packet_loss_pct",
        "satellite_packet_loss_pct",
        "anomaly_score",
        "trust_score"
    ]
    
    # Check which columns exist in the dataframe
    available_columns = [col for col in top_20_columns if col in most_anomalous.columns]
    
    # Create top 20 dataframe with model tracking columns
    top_20_df = most_anomalous[available_columns].copy()
    top_20_df.insert(0, "model_type", "IsolationForest")
    top_20_df.insert(1, "n_estimators", n_estimators)
    
    top_20_path = os.path.join(results_dir, f"top_20_anomalies_IsolationForest_{n_estimators}.csv")
    top_20_df.to_csv(top_20_path, index=False)
    print(f"Saved top 20 anomalies to: {top_20_path}")

    print(f"\n{'='*70}")
    print(f"Research output saved to '{results_dir}/' directory")
    print(f"{'='*70}")