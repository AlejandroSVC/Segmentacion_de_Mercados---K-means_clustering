# Segmentación del mercado
## Uso del Análisis de Conglomerados de k-Medias y Análisis Discriminante

## Python

![Customer_segmentation](docs/assets/images/Customer_segmentation.jpg)

### Importar bibliotecas
```
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```
### Cargar los datos
```
import os
os.chdir('C:/Users/Alejandro/Documents/')
data = pd.read_csv('Customer_segmentation.csv')
data.info()
```
### Estandarizar los datos (importante para K-Means)
```
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```
### Agrupamiento de K-Medias (optimice n_clusters si es necesario)
```
n_clusters = 4             # Ajuste según el método del codo o la puntuación de silueta
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
```
### Agregar etiquetas de clúster a los datos originales
```
data['cluster'] = clusters
```
### Evaluar la calidad de la agrupación
```
silhouette_avg = silhouette_score(scaled_data, clusters)
print(f"Silhouette Score (clustering quality): {silhouette_avg:.2f}")
```
### Análisis discriminante (LDA) para clasificar variables
```
lda = LinearDiscriminantAnalysis()
lda.fit(scaled_data, clusters)
```
### Mostrar las estadísticas del Análisis Discriminante
```
print("=== DISCRIMINANT ANALYSIS RESULTS ===")
print("\nEigenvalues (explained variance ratio) for each discriminant function:")
print(lda.explained_variance_ratio_)
```
### Extraer y clasificar variables discriminantes
```
discriminant_power = pd.DataFrame({
    'variable': data.columns[:-1],  # Exclude 'cluster' column
    'discriminant_weight': np.abs(lda.coef_[0])  # Magnitud de los coefficients LDA
}).sort_values('discriminant_weight', ascending=False)

print("\nMost Discriminant Variables (ranked):")
print(discriminant_power)
```
### Calcular medias de grupo para las 4 principales variables discriminantes
```
top_4_vars  = discriminant_power['variable'].head(4).tolist()
group_means = data.groupby('cluster')[top_4_vars].mean()

print("\nGroup Means (Principales 4 Variables Discriminantes):")
print(group_means)
```
## RESULTADO:

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

C = cluster

v1 = AcceptedCmp1

v2 = NumStorePurchases

v3 = MntMeatProducts

v4 = AcceptedCmp2

C            v1             v2         v3          v4

0            0.00           3.11        27.44      0.00

1            0.00           9.18       413.05      0.00

2            0.00           7.41       149.04      0.00

3            0.83           9.50       577.50      0.33
