import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. LOAD DATA
df = pd.read_csv('heart.csv') # Dataset from Kaggle

# Preprocessing: Convert categories to numeric
df_numeric = pd.get_dummies(df, drop_first=True)
X = df_numeric.drop('HeartDisease', axis=1)

# Scale features (Critical for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. ISOLATION FOREST (Outlier Removal)
# contamination=0.05 assumes 5% of data are anomalies/noise
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_scaled)

# Filter out the -1 values (anomalies)
X_cleaned = X_scaled[outliers == 1]
y_cleaned = df_numeric['HeartDisease'][outliers == 1]

# 3. FUZZY C-MEANS CLUSTERING
# FCM expects data in (features, samples) shape, so we transpose
data_to_cluster = X_cleaned.T
n_clusters = 2  # Healthy vs Heart Failure

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_to_cluster, c=n_clusters, m=2, error=0.005, maxiter=1000
)

# 4. PREDICTION & INTERPRETATION
# Get the cluster with the highest membership for each point
cluster_membership = np.argmax(u, axis=0)

# Check which cluster corresponds to 'Heart Disease' by comparing with labels
# Note: Cluster 0 or 1 is arbitrary; we map them to the real labels
from scipy.stats import mode
labels = np.zeros_like(cluster_membership)
for i in range(n_clusters):
    mask = (cluster_membership == i)
    labels[mask] = mode(y_cleaned[mask], keepdims=True)[0]

# 5. EVALUATION
from sklearn.metrics import classification_report, accuracy_score
print("Model Evaluation (FCM Clusters vs Real Labels):")
print(classification_report(y_cleaned, labels))
print(f"Accuracy: {accuracy_score(y_cleaned, labels):.2f}")

# Example of Fuzzy Membership for the first 5 patients
print("\nFuzzy Membership Degrees (First 5 patients):")
# Each column is a patient; rows are Cluster 0 and Cluster 1
print(pd.DataFrame(u[:, :5], index=['Cluster_0_Prob', 'Cluster_1_Prob']).T)
