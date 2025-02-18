import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import seaborn as sns
#from google.colab import files

def calculate_rfm(data):
  """
  Calculates Recency, Frequency, and Monetary values for customer transactions.

  Args:
    data: A pandas DataFrame with at least the following columns:
      - Customer_ID: Unique identifier for each customer.
      - Date: Date of the transaction.
      - Sales_Amount: The amount of the transaction.

  Returns:
    A pandas DataFrame with RFM values for each customer.
  """

  # Convert Date column to datetime if it's not already
  data['Date'] = pd.to_datetime(data['Date'])

  # Calculate Recency
  recent_date = data['Date'].max()
  recency_df = data.groupby('Customer_ID')['Date'].max().reset_index()
  recency_df['Recency'] = (recent_date - recency_df['Date']).dt.days
  recency_df = recency_df[['Customer_ID', 'Recency']]

  # Calculate Frequency
  frequency_df = data.groupby('Customer_ID')['Date'].count().reset_index()
  frequency_df.rename(columns={'Date': 'Frequency'}, inplace=True)

  # Calculate Monetary
  monetary_df = data.groupby('Customer_ID')['Sales_Amount'].sum().reset_index()
  monetary_df.rename(columns={'Sales_Amount': 'Monetary'}, inplace=True)

  # Merge RFM values
  rfm_df = pd.merge(recency_df, frequency_df, on='Customer_ID')
  rfm_df = pd.merge(rfm_df, monetary_df, on='Customer_ID')

  return rfm_df


# Load the datasets
transactions = pd.read_excel("Orders2.0.xlsx")
customers = pd.read_excel("Customers.xlsx")

# Merge the datasets on 'Customer_ID'
data = pd.merge(transactions, customers, on='Customer_ID', how='left')

# Print some info to check the merge
print(data.info())

rfm_df = calculate_rfm(data)

# Select RFM columns for clustering
rfm_data = rfm_df[['Recency', 'Frequency', 'Monetary']]

# Standardize the data (important for KMeans)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)

# Find optimal number of clusters (using Elbow method)
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
optimal_clusters = knee.elbow
print(f"Optimal number of clusters: {optimal_clusters}")

# Apply KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Analyze the clusters
print(rfm_df.groupby('Cluster').agg({'Recency': 'mean',
                                     'Frequency': 'mean',
                                     'Monetary': 'mean'}))

# (Optional) Visualize clusters (if you have 2D or 3D data)
import matplotlib.pyplot as plt
plt.scatter(rfm_scaled[:, 0], rfm_scaled[:, 1], c=rfm_df['Cluster'], cmap='viridis')
plt.savefig("Plots/CustomerClusters.jpeg", dpi=300)
plt.show()

# 2.1.3 Exploratory Data Analysis (EDA)

# 2.1.3.1 Descriptive Statistics
print("Descriptive Statistics:")
print(rfm_df.describe())  # Provides summary statistics for all numerical columns


# 2.1.3.2 Data Visualization
plt.figure(figsize=(12, 4))

# Histograms for RFM
plt.subplot(1, 3, 1)
sns.histplot(rfm_df['Recency'], kde=True)
plt.title('Recency Distribution')

plt.subplot(1, 3, 2)
sns.histplot(rfm_df['Frequency'], kde=True)  # Using sns.histplot
plt.title('Frequency Distribution')

plt.subplot(1, 3, 3)
sns.histplot(rfm_df['Monetary'], kde=True)
plt.title('Monetary Distribution')

plt.tight_layout()
plt.savefig("Plots/RFM.jpeg", dpi=300)
plt.show()

# Scatter plots (example - Recency vs. Frequency)
plt.figure(figsize=(6, 6))
sns.scatterplot(x='Recency', y='Frequency', data=rfm_df)
plt.title('Recency vs. Frequency')
plt.show()


# Box plots for outlier detection (example - Monetary)
plt.figure(figsize=(6, 6))
sns.boxplot(y='Monetary', data=rfm_df)
plt.title('Monetary Value Box Plot')
plt.savefig("Plots/MonetaryValueBoxPlot.jpeg", dpi=300)
plt.show()


# 2.1.3.3 Product Category Affinity
category_affinity = data.groupby('Customer_ID')['SKU_Category'].value_counts(normalize=True).unstack()
category_affinity = category_affinity.fillna(0) # Fill NaNs with 0 for customers who haven't purchased from all categories
rfm_df = rfm_df.merge(category_affinity, on='Customer_ID', how='left')  # Add to rfm_df

# Print some info or save the data
print(rfm_df.info())
# rfm_with_affinity.to_csv("rfm_with_product_affinity.csv", index=False)



# 2.1.3.4 Demographic Analysis (if demographics are available)
# Merge with demographic data if you haven't already
if 'GENDER' not in rfm_df.columns:  # Check if already merged (from 2.1.1)
    rfm_df = rfm_df.merge(data[['Customer_ID', 'GENDER', 'AGE', 'GEOGRAPHY']], on='Customer_ID', how='left')

# Example: Average Monetary Value by Gender
plt.figure(figsize=(8, 6))
sns.barplot(x='GENDER', y='Monetary', data=rfm_df)
plt.title('Average Monetary Value by Gender')
plt.savefig("Plots/AverageMonetaryValueByGender.jpeg", dpi=300)
plt.show()



# Example: Distribution of Age by Geography
plt.figure(figsize=(10,6))
sns.boxplot(x='GEOGRAPHY', y='AGE', data=rfm_df)
plt.title('Distribution of Age by Geography')
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.savefig("Plots/DistributionOfAgeByGeography.jpeg", dpi=300)
plt.show()


# 2.1.4 Customer Segmentation

# 2.1.4.1 Feature Selection for Clustering
features_for_clustering = ['Recency', 'Frequency', 'Monetary']  # Start with RFM
# Add relevant category affinity or demographic features (if found significant during EDA)
# Example:
# features_for_clustering.extend(['X52', '2ML', '0H2', 'AGE']) # Add specific categories and age


# Select the chosen features
rfm_data = rfm_df[features_for_clustering]


# 2.1.4.2 Standardize Data (important for KMeans)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)



# 2.1.4.3 Determine Optimal Number of Clusters (Elbow Method)
wcss = []
for i in range(1, 11):  # Test clusters from 1 to 10
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
optimal_clusters = knee.elbow
print(f"Optimal number of clusters: {optimal_clusters}")


# 2.1.4.4 Apply KMeans Clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)



# 2.1.4.5 Profile and Name Customer Segments
# Analyze the clusters
print(rfm_df.groupby('Cluster').agg({'Recency': 'mean',
                                     'Frequency': 'mean',
                                     'Monetary': 'mean'}))

# Add more detailed profiling based on demographics, category affinities, etc.

# Example Segment Naming (adapt based on your results):
if optimal_clusters == 4:  # Example: 4 clusters
  cluster_labels = {
      0: "Champions",
      1: "Loyal Customers",
      2: "Potential Loyalist",
      3: "At Risk"
  }
elif optimal_clusters == 3: #Example: 3 clusters
  cluster_labels = {
      0: "Champions",
      1: "Loyal Customers",
      2: "At Risk"
  }
#... add more label conditions if needed for other optimal_cluster numbers
else:
  cluster_labels = {i: f"Cluster {i}" for i in range(optimal_clusters)} #Generic cluster names if there are not specific labels defined


rfm_df['Segment'] = rfm_df['Cluster'].map(cluster_labels)

print(rfm_df.head())


# (Optional) Save the segmented data
rfm_df.to_csv('rfm_segmented_data.csv', index=False)

# 2.1.4.6 Visualizations

# Visualize RFM distributions within each cluster
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x='Segment', y='Recency', data=rfm_df)
plt.title('Recency by Segment')

plt.subplot(1, 3, 2)
sns.boxplot(x='Segment', y='Frequency', data=rfm_df)
plt.title('Frequency by Segment')

plt.subplot(1, 3, 3)
sns.boxplot(x='Segment', y='Monetary', data=rfm_df)
plt.title('Monetary by Segment')


plt.tight_layout()

plt.savefig("Plots/SegmentNaming.jpeg", dpi=300)

plt.show()



# Visualize segment sizes (optional)
segment_counts = rfm_df['Segment'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
segment_counts.plot(kind='bar')
plt.title('Segment Sizes')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.show()

# Scatter plot of Recency vs. Frequency colored by segment
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Recency', y='Frequency', hue='Segment', data=rfm_df, palette='viridis')  # Use a color palette
plt.title('Recency vs. Frequency (Colored by Segment)')
plt.show()




# 3D scatter plot (if using 3 features for clustering)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = rfm_scaled[:, 0]  # Recency (scaled)
y = rfm_scaled[:, 1]  # Frequency (scaled)
z = rfm_scaled[:, 2]  # Monetary (scaled)

ax.scatter(x, y, z, c=rfm_df['Cluster'], cmap='viridis', marker='o')

ax.set_xlabel('Recency (Scaled)')
ax.set_ylabel('Frequency (Scaled)')
ax.set_zlabel('Monetary (Scaled)')
plt.title('RFM Clusters (3D)')
plt.savefig("Plots/RFMClusters3D.jpeg", dpi=300)
plt.show()


# Pairplot (if you have more features and want to see all pairwise relationships)
sns.pairplot(rfm_df[features_for_clustering + ['Segment']], hue='Segment')
plt.show()


# 2.1.4.5 Product Category Affinity


# 1. Top N SKUs across all customers
top_n = 10  # Example: Top 10 SKUs

top_skus = category_affinity.sum().sort_values(ascending=False).head(top_n)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_skus.index, y=top_skus.values)
plt.title(f"Top {top_n} Most Popular SKUs")
plt.xlabel("SKU")
plt.ylabel("Total Affinity (Normalized Purchase Frequency)")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.tight_layout()
plt.savefig("Plots/Top10SKUsAcrossAllCustomers.jpeg", dpi=300)
plt.show()



# 2. Heatmap of Product Affinity for a segment
segment_to_analyze = 'Champions'
segment_affinity = rfm_df[rfm_df['Segment'] == segment_to_analyze][category_affinity.columns].mean()
plt.figure(figsize=(12, 8))
sns.heatmap(segment_affinity.values.reshape(1,-1),  # Reshape for single segment
            annot=True, cmap="YlGnBu", fmt=".2f",
            xticklabels=segment_affinity.index, yticklabels=[segment_to_analyze])
plt.title(f"Average Product Affinity for Segment: {segment_to_analyze}")
plt.tight_layout()
plt.savefig("Plots/HeatmapOfProductAffinityChampions.jpeg", dpi=300)
plt.show()



# 3. Top SKUs for a specific segment
top_n_segment = 5

segment_affinity = rfm_df[rfm_df['Segment'] == segment_to_analyze][category_affinity.columns].mean().sort_values(ascending = False)
plt.figure(figsize=(10, 6))
sns.barplot(x=segment_affinity.head(top_n_segment).index, y=segment_affinity.head(top_n_segment).values)
plt.title(f"Top {top_n_segment} SKUs for Segment: {segment_to_analyze}")
plt.xlabel("SKU")
plt.ylabel("Average Affinity")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.tight_layout()
plt.savefig("Plots/TopSKUsForChampions.jpeg", dpi=300)
plt.show()
