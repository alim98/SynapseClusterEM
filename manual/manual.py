import pandas as pd
import numpy as np
import gower
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# ---- Step 1: Load Files ----
folder = 'manual/'
metadata_file = folder + "metadata.csv"
data_file = folder + "manual_synapse_features_60synapses.csv"

# ---- Step 2: Load Metadata ----
metadata = pd.read_csv(metadata_file, delimiter=";", header=None)
feature_names = metadata.iloc[:, 0].dropna().tolist()
feature_types = metadata.iloc[:, 1].dropna().tolist()
categories_list = metadata.iloc[:, 2:].apply(lambda x: x.dropna().tolist(), axis=1).tolist()

# ---- Step 2: Load and Format Data ----
raw_data = pd.read_csv(data_file, delimiter=";", header=None)
multi_index = pd.MultiIndex.from_arrays(raw_data.iloc[:2].values, names=["SynapseID", "BBox"])
data = raw_data.iloc[2:]
data.columns = multi_index
data = data.T
data.columns = feature_names

# Separate features
nominal_features = [feature_names[i] for i, ftype in enumerate(feature_types) if ftype == "nominal"]
ordinal_features = [feature_names[i] for i, ftype in enumerate(feature_types) if ftype == "ordinal"]

# Create a dictionary of ordinal feature name -> list of categories
ordinal_categories = {
    feature_names[i]: categories_list[i]
    for i, ftype in enumerate(feature_types)
    if ftype == "ordinal"
}

# ---- Step 5: Encode Nominal Features ----
encoder_nominal = OneHotEncoder(sparse_output=False, drop="first")
df_nominal = pd.DataFrame(index=data.index)

for feature in nominal_features:
    if not data[feature].isnull().all():
        encoded = encoder_nominal.fit_transform(data[[feature]])
        col_names = encoder_nominal.get_feature_names_out([feature])
        df_nominal = pd.concat([df_nominal, pd.DataFrame(encoded, index=data.index, columns=col_names)], axis=1)

# ---- Step 6: Encode Ordinal Features ----
df_ordinal = pd.DataFrame(index=data.index)
if ordinal_features:
    categories = [ordinal_categories[f] for f in ordinal_features]
    encoder_ordinal = OrdinalEncoder(categories=categories)
    encoded = encoder_ordinal.fit_transform(data[ordinal_features])
    df_ordinal = pd.DataFrame(encoded, index=data.index, columns=ordinal_features)

# ---- Step 7: Combine Encoded Data ----
df_encoded = pd.concat([df_nominal, df_ordinal], axis=1)

# ---- Step 8: Compute Gower Distance ----
distance_matrix = gower.gower_matrix(df_encoded)

# ---- Step 9: Hierarchical Clustering ----
linkage_matrix = linkage(distance_matrix, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=df_encoded.index, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram")
plt.tight_layout()
# plt.show()

# ---- Step 10: Assign Clusters ----
num_clusters = 2
clusters = fcluster(linkage_matrix, num_clusters, criterion="maxclust")
df_encoded["Cluster"] = clusters

# ---- Step 11: Save Results ----
df_encoded.to_csv("clustered_samples.csv")
print("Clustering complete. Results saved to clustered_samples.csv.")
print(feature_names)
data.head()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---- Step 1: Standardize the Encoded Data ----
scaler = StandardScaler()
df_encoded.columns = df_encoded.columns.astype(str)
X_scaled = scaler.fit_transform(df_encoded.drop(columns=["Cluster"]))  # Exclude cluster from PCA

# ---- Step 2: Apply PCA ----
n_components = 4  # Or however many you want
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(X_scaled)

# Create DataFrame with all components
pca_columns = [f"PC{i+1}" for i in range(n_components)]
df_pca = pd.DataFrame(pca_result, columns=pca_columns, index=df_encoded.index)
df_pca["Cluster"] = df_encoded["Cluster"]

# ---- Step 3: Plot only the first two components ----
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Cluster", palette="viridis", s=50, edgecolor="k")

plt.title("PCA Visualization of Clusters (First 2 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.grid(True)
# plt.show()
