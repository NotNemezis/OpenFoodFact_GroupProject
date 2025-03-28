import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors



### Alexandre Code 
# class DataPreprocessor:


#     def pre_processing(self):
#         """
#         Effectue les étapes de prétraitement suivantes sur le DataFrame :
#         1. Sélectionne les colonnes avec '_100g' dans leur nom.
#         2. Supprime les colonnes fortement corrélées (seuil > 0.8).
#         3. Supprime les lignes avec des valeurs manquantes.
#         4. Standardise les données (normalisation).

#         Retour :
#             pd.DataFrame : Les données normalisées et traitées.
#         """
#         if self.df.empty:
#             print("DataFrame is empty. Please check the file path and try again.")
#             return self.df

#         # Sélectionner les colonnes '_100g'
#         col100g = [col for col in self.df.columns if "_100g" in col]
#         data_100g = self.df[col100g]

#         # Calcul de la matrice de corrélation
#         corr_matrix = data_100g.corr().abs()

#         # Déterminer les colonnes fortement corrélées (seuil > 0.8)
#         upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#         to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

#         # Supprimer les colonnes redondantes
#         data_100g = data_100g.drop(columns=to_drop)

#         # Supprimer les lignes contenant des valeurs manquantes
#         data_100g = data_100g.dropna()

#         return data_100g


# class DBSCANAnalyzer:
#     """
#     Classe pour appliquer DBSCAN sur les données prétraitées.
#     """

#     def __init__(self, data):
#         """
#         Initialise la classe avec les données prétraitées.

#         Arguments :
#             data (pd.DataFrame) : Les données prétraitées.
#         """
#         if data.empty:
#             raise ValueError("Data is empty. Please provide a valid DataFrame.")
#         self.data = data

#     def apply_dbscan(self, eps=5, min_samples=5):
#         """
#         Applique DBSCAN après normalisation des données et traitement des outliers.

#         Arguments :
#             eps (float) : Distance maximale entre deux échantillons pour qu'ils soient considérés comme voisins.
#             min_samples (int) : Nombre minimum d'échantillons dans un voisinage pour qu'un point soit considéré comme un noyau.

#         Retour :
#             pd.DataFrame : Le DataFrame avec les clusters ajoutés.
#         """
#         # Standardisation des données
#         scaler = StandardScaler()
#         scaled_data = scaler.fit_transform(self.data)

#         # Appliquer t-SNE pour réduire à 2D
#         tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#         X_tsne = tsne.fit_transform(scaled_data)

#         # Appliquer DBSCAN sur les résultats de t-SNE
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
#         clusters = dbscan.fit_predict(X_tsne)

#         # Ajouter les clusters au DataFrame
#         self.data["tsne1"], self.data["tsne2"] = X_tsne[:, 0], X_tsne[:, 1]
#         self.data["cluster"] = clusters

#         return self.data

#     def get_kdist_plot(self, k):
#         """
#         Affiche le graphique des distances k-distances pour déterminer le bon paramètre eps pour DBSCAN.

#         Arguments :
#             k (int) : Nombre de voisins à utiliser pour le calcul des distances.
#         """
#         # Standardisation des données
#         scaler = StandardScaler()
#         scaled_data = scaler.fit_transform(self.data)

#         nbrs = NearestNeighbors(n_neighbors=k).fit(scaled_data)
#         distances, _ = nbrs.kneighbors(scaled_data)

#         distances = np.sort(distances, axis=0)
#         distances = distances[:, k-1]

#         plt.figure(figsize=(8, 8))
#         plt.plot(distances)
#         plt.xlabel('Points/Objects in the dataset')
#         plt.ylabel(f'Sorted {k}-nearest neighbor distance')
#         plt.grid(True, linestyle="--", color='black', alpha=0.4)
#         plt.show()