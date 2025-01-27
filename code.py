import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

customers_path = "Customers.csv"
products_path = "Products.csv"
transactions_path = "Transactions.csv"

customers = pd.read_csv(customers_path)
products = pd.read_csv(products_path)
transactions = pd.read_csv(transactions_path)

print("Customers Data:")
print(customers.info())
print(customers.describe())

print("Products Data:")
print(products.info())
print(products.describe())

print("Transactions Data:")
print(transactions.info())
print(transactions.describe())

merged_data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

plt.figure(figsize=(10, 6))
sns.countplot(data=customers, x="Region", order=customers["Region"].value_counts().index)
plt.title("Customer Distribution by Region")
plt.xticks(rotation=45)
plt.savefig("customer_distribution_by_region.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(data=merged_data, x="Category", y="TotalValue", estimator=sum, ci=None, order=merged_data.groupby("Category")["TotalValue"].sum().sort_values(ascending=False).index)
plt.title("Revenue by Product Category")
plt.xticks(rotation=45)
plt.savefig("revenue_by_product_category.png")
plt.close()

customer_features = merged_data.groupby("CustomerID")["TotalValue"].sum().reset_index()
customer_features = customer_features.merge(customers, on="CustomerID")

feature_matrix = customer_features[["TotalValue"]]
scaler = StandardScaler()
feature_matrix_scaled = scaler.fit_transform(feature_matrix)
similarity_matrix = cosine_similarity(feature_matrix_scaled)

lookalike_results = {}
for i in range(20):
    customer_id = customer_features.iloc[i]["CustomerID"]
    similarities = list(enumerate(similarity_matrix[i]))
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[1:4]
    lookalike_results[customer_id] = [(customer_features.iloc[idx]["CustomerID"], score) for idx, score in sorted_similarities]

lookalike_df = pd.DataFrame({"CustomerID": list(lookalike_results.keys()), "Lookalikes": [str(v) for v in lookalike_results.values()]})
lookalike_df.to_csv("Lookalike.csv", index=False)

clustering_features = merged_data.groupby("CustomerID").agg({"TotalValue": "sum", "Quantity": "sum"}).reset_index()
scaler = StandardScaler()
clustering_scaled = scaler.fit_transform(clustering_features.drop(columns=["CustomerID"]))

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(clustering_scaled)
clustering_features["Cluster"] = kmeans.labels_

from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(clustering_scaled, kmeans.labels_)
print(f"Davies-Bouldin Index: {db_index}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=clustering_scaled[:, 0], y=clustering_scaled[:, 1], hue=kmeans.labels_, palette="viridis")
plt.title("Customer Segmentation Clusters")
plt.savefig("customer_segmentation_clusters.png")
plt.close()

clustering_features.to_csv("Clustering_Results.csv", index=False)

print("EDA, Lookalike Model, and Clustering Completed.")
