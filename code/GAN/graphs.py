import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs("compare", exist_ok=True)

# Define file paths for each model
file_paths = {
    "CNN": "CNN/txt/evaluation_metrics_media_CNN.txt",
    # "INCEPTION": "INCEPTION/txt/evaluation_metrics_media_INCEPTION.txt",
    "XCEPTION": "XCEPTION/txt/evaluation_metrics_media_XCEPTION.txt"
}

# Initialize a dictionary to store metric data
metrics_data = {
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "Validation Accuracy": [],
    "Validation Loss": [],
    "Validation Precision": [],
    "Validation Recall": []
}

# Function to read metrics from each file
def read_metrics(file_path):
    metrics = {}
    with open(file_path, "r") as f:
        for line in f:
            metric, value = line.strip().split(":")
            metric = metric.strip().title()  # Standardize metric names
            value = float(value.strip().replace("%", ""))  # Convert value to float, remove "%"
            metrics[metric] = value
    return metrics

# Populate metrics_data with values from each model
for model_name, file_path in file_paths.items():
    model_metrics = read_metrics(file_path)
    for metric in metrics_data.keys():
        metrics_data[metric].append(model_metrics.get(metric, 0))

# Models list
models = list(file_paths.keys())

# 1. Individual Bar Graphs for Each Metric
for metric, values in metrics_data.items():
    plt.figure(figsize=(8, 6))
    plt.bar(models, values, color=['blue', 'green', 'purple'])
    plt.title(f"{metric} Comparison Across Models")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(models)
    plt.savefig(f"compare/{metric.replace(' ', '_').lower()}_comparison.png")
    plt.close()

# 2. Line Graph for All Metrics Across Models
plt.figure(figsize=(10, 8))
for metric, values in metrics_data.items():
    plt.plot(models, values, marker='o', label=metric)
plt.title("Metrics Comparison Across Models")
plt.xlabel("Model")
plt.ylabel("Value")
plt.legend()
plt.savefig("compare/metrics_comparison_line_graph.png")
plt.close()

# 3. Grouped Bar Graph for Training vs Validation Metrics
train_metrics = ["Accuracy", "Precision", "Recall"]
val_metrics = ["Validation Accuracy", "Validation Precision", "Validation Recall"]

x = np.arange(len(models))
width = 0.35

for i, metric_pair in enumerate(zip(train_metrics, val_metrics)):
    train_metric, val_metric = metric_pair
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, metrics_data[train_metric], width, label=f"Train {train_metric}", color='blue')
    ax.bar(x + width / 2, metrics_data[val_metric], width, label=f"Validation {val_metric}", color='orange')
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Value")
    ax.set_title(f"Training vs Validation {train_metric}/{val_metric} Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.savefig(f"compare/{train_metric.replace(' ', '_').lower()}_vs_{val_metric.replace(' ', '_').lower()}.png")
    plt.close()

# 4. Heatmap of All Metrics Across Models
# Convert metrics_data to a 2D array suitable for heatmap
heatmap_data = np.array([metrics_data[metric] for metric in metrics_data.keys()])
heatmap_labels = list(metrics_data.keys())

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=models, yticklabels=heatmap_labels)
plt.title("Metrics Heatmap Across Models")
plt.xlabel("Model")
plt.ylabel("Metric")
plt.savefig("compare/metrics_heatmap.png")
plt.close()

# 5. Radar Chart for Each Model
metrics_for_radar = ["Accuracy", "Precision", "Recall", "Validation Accuracy", "Validation Precision", "Validation Recall"]
# Generate angles for the radar chart based on the number of metrics
angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()

for i, model in enumerate(models):
    # Retrieve values for the current model and metrics
    values = [metrics_data[metric][i] for metric in metrics_for_radar]
    # Close the loop by appending the first value to the end of the list
    values += values[:1]
    # Close the loop for angles by appending the first angle
    angles_with_closure = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles_with_closure, values, color='blue', linewidth=2, label=model)
    ax.fill(angles_with_closure, values, color='blue', alpha=0.25)
    # Set y-gridlines (concentric circles) and labels on them
    ax.set_rgrids(np.arange(0.1, 1.1, step=0.1), labels=[f"{x*100}%" for x in np.arange(0.1, 1.1, step=0.1)], angle=225)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(metrics_for_radar)
    ax.set_title(f"{model} Metrics Radar Chart")

    # Display the metric values on the chart
    for j, (angle, value) in enumerate(zip(angles, values)):
        ax.text(angle, value, f"{value:.2f}", ha='center', va='center', color='black', size=10)

    plt.legend(loc="upper right")
    plt.savefig(f"compare/{model}_radar_chart.png")
    plt.close()

