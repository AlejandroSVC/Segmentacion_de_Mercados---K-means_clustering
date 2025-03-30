# Market Segmentation
## Using k-means cluster analysis and discriminant analysis with Python

![Customer_segmentation](docs/assets/images/Customer_segmentation.jpg)

### Libraries
```
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```
### Load data
```
import os
os.chdir('C:/Users/Alejandro/Documents/')
data = pd.read_csv('Customer_segmentation.csv')
data.info()
```
### Standardize data (important for K-Means)
```
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```
### K-Means Clustering (optimize n_clusters if needed)
```
n_clusters = 4  # Adjust based on elbow method or silhouette score
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
```
### Add cluster labels to original data
```
data['cluster'] = clusters
```
### Evaluate clustering quality
```
silhouette_avg = silhouette_score(scaled_data, clusters)
print(f"Silhouette Score (clustering quality): {silhouette_avg:.2f}")
```
### Discriminant Analysis (LDA) to rank variables
```
lda = LinearDiscriminantAnalysis()
lda.fit(scaled_data, clusters)
```
### Print discriminant analysis statistics
```
print("=== DISCRIMINANT ANALYSIS RESULTS ===")
print("\nEigenvalues (explained variance ratio) for each discriminant function:")
print(lda.explained_variance_ratio_)
```
### Extract & rank discriminant variables
```
discriminant_power = pd.DataFrame({
    'variable': data.columns[:-1],  # Exclude 'cluster' column
    'discriminant_weight': np.abs(lda.coef_[0])  # Magnitude of LDA coefficients
}).sort_values('discriminant_weight', ascending=False)

print("\nMost Discriminant Variables (ranked):")
print(discriminant_power)
```
### Compute group means for top 4 discriminant variables
```
top_4_vars = discriminant_power['variable'].head(4).tolist()
group_means = data.groupby('cluster')[top_4_vars].mean()

print("\nGroup Means (Top 4 Discriminant Variables):")
print(group_means)
```
OUTPUT

=== DISCRIMINANT ANALYSIS RESULTS ===

Eigenvalues (explained variance ratio) for each discriminant function:

[0.69828274 0.211259   0.09045826]

Ten Most Discriminant Variables (ranked):

variable  discriminant_weight

19. AcceptedCmp1: 4.142822

14. NumStorePurchases: 3.630316

7. MntMeatProducts: 3.251910

20. AcceptedCmp2: 2.236848

0. Year_Birth:  1.699472

18. AcceptedCmp5: 1.295870

8. MntFishProducts: 0.985022

17. AcceptedCmp4: 0.918370

12. NumWebPurchases: 0.901000

2. Kidhome: 0.880786

Group Means (Top 4 Discriminant Variables):

         AcceptedCmp1  NumStorePurchases  MntMeatProducts  AcceptedCmp2
cluster  


0            0.000000           3.111111        27.444444      0.000000


1            0.000000           9.181818       413.045455      0.000000

2            0.000000           7.407407       149.037037      0.000000

3            0.833333           9.500000       577.500000      0.333333
