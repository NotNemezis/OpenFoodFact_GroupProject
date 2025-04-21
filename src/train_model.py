"""
Module pour l'Analyse en Composantes Principales (ACP) et K-Means sur le dataset Open Food Facts.

Contient les classes `DataPreprocessor` et `KMeansAnalyzer` permettant de :
- Nettoyer le dataset en supprimant les colonnes non pertinentes.
- Éliminer les doublons afin d'assurer la qualité des données.
- Supprimer les colonnes contenant un taux trop élevé de valeurs manquantes.
- Appliquer l'ACP pour réduire la dimensionnalité des données.
- Appliquer K-Means pour le clustering.
- Visualiser les résultats de l'ACP et du clustering.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

class DataPreprocessor:
    """
    Classe pour le prétraitement des données du dataset Open Food Facts.
    """

    def __init__(self, file_path="../openfoodsfacts_sample.csv", nrows=30000):
        """
        Initialise la classe en chargeant un échantillon du dataset Open Food Facts.

        Arguments :
            file_path (str) : Chemin du fichier CSV à charger.
            nrows (int) : Nombre de lignes à charger (par défaut 10 000).
        """
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        try:
            self.df = pd.read_csv(
                file_path,
                sep="\t",
                on_bad_lines='skip',
                nrows=nrows,
                low_memory=False
            )
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            self.df = pd.DataFrame()

class KMeansAnalyzer:
    """
    Classe pour appliquer K-Means sur les données prétraitées.
    """

    def __init__(self, data):
        """
        Initialise la classe avec les données prétraitées.

        Arguments :
            data (pd.DataFrame) : Les données prétraitées.
        """
        if data.empty:
            raise ValueError("Data is empty. Please provide a valid DataFrame.")
        self.data = data

    def kmeans(self, n_clusters, threshold=1.5):
        """
        Applique K-Means après normalisation des données.

        Arguments :
            n_clusters (int) : Nombre de clusters (par défaut : 3).

        Retour :
            pd.DataFrame : Le DataFrame avec une nouvelle colonne 'cluster' avec des labels séquentiels.
        """
        # Calcul des bornes pour la détection des outliers
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1

        # Définir les bornes inférieures et supérieures pour chaque colonne
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Filtrer les données en supprimant les valeurs anormales (outliers)
        self.data = self.data[~((self.data < lower_bound) | (self.data > upper_bound)).any(axis=1)]

        # Standardisation des données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        # Appliquer KMeans avec une initialisation multiple des centroids (n_init)
        kmeans = KMeans(n_clusters=n_clusters, random_state=25)

        # Assigner les clusters
        self.data.loc[:, "cluster"] = kmeans.fit_predict(scaled_data)

        # Vérification de l'inertie et des résultats
        print("Inertie du modèle K-Means : ", kmeans.inertia_)

        # Vérifier les labels et la distribution des points par cluster
        print(f"Cluster labels: {sorted(set(self.data['cluster']))}")
        print(f"Distribution des points par cluster: \n{self.data['cluster'].value_counts()}")

        return self.data

    def find_clusters_elbow(self):
        """
        Utilise la méthode du coude pour déterminer le nombre optimal de clusters.

        Retour :
            None
        """
        inertias = []

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)

        plt.plot(range(1, 11), inertias, marker='o')
        plt.title('Elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

    def plot_kmeans_clusters(self, threshold=1.5):
        """
        Réduit les données à 2 dimensions avec PCA et affiche les clusters.

        Arguments :
            threshold (float) : Seuil pour la détection des outliers basé sur l'IQR. Par défaut 1.5.
        """
        # Calcul des quartiles et de l'IQR pour chaque colonne
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1

        # Définir les bornes inférieures et supérieures pour chaque colonne
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Filtrer les données pour éliminer les outliers
        self.data = self.data[~((self.data < lower_bound) | (self.data > upper_bound)).any(axis=1)]

        # Standardisation des données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        # Appliquer l'ACP pour réduire les dimensions à 2
        pca = PCA(n_components=2)
        pca_scores = pca.fit_transform(scaled_data)  # Projection des données dans l'espace 2D

        # Ajouter les résultats de l'ACP aux données pour les utiliser dans le graphique
        self.data['pca1'] = pca_scores[:, 0]
        self.data['pca2'] = pca_scores[:, 1]

        # Créer le graphique
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.data['pca1'], y=self.data['pca2'], hue=self.data["cluster"], palette="Set1", alpha=0.7)

        # Ajouter des lignes de référence pour l'axe des X et Y
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)

        # Ajouter des labels et un titre
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.title("Nuage de points des données après ACP et K-Means")
        plt.show()

    def plot_kmeans_clusters_tsne(self, threshold=1.5, perplexity=15, n_iter=800):
        """
        Réduit les données à 2 dimensions avec t-SNE et affiche les clusters.

        Arguments :
            threshold (float) : Seuil pour la détection des outliers basé sur l'IQR. Par défaut 1.5.
            perplexity (int) : Perplexité pour t-SNE. Par défaut 30.
            n_iter (int) : Nombre d'itérations pour t-SNE. Par défaut 1000.
        """
        # Calcul des quartiles et de l'IQR pour chaque colonne
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1

        # Définir les bornes inférieures et supérieures pour chaque colonne
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Filtrer les données pour éliminer les outliers
        self.data = self.data[~((self.data < lower_bound) | (self.data > upper_bound)).any(axis=1)]

        # Standardisation des données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        # Appliquer t-SNE pour réduire les dimensions à 2
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=25)
        tsne_scores = tsne.fit_transform(scaled_data)  # Projection des données dans l'espace 2D

        # Ajouter les résultats de t-SNE aux données pour les utiliser dans le graphique
        self.data['tsne1'] = tsne_scores[:, 0]
        self.data['tsne2'] = tsne_scores[:, 1]

        # Créer le graphique
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.data['tsne1'], y=self.data['tsne2'], hue=self.data["cluster"], palette="Set1", alpha=0.7)

        # Ajouter des lignes de référence pour l'axe des X et Y
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
class DBSCANAnalyzer:
    """
    Classe pour appliquer DBSCAN sur les données prétraitées.
    """

    def __init__(self, data):
        """
        Initialise la classe avec les données prétraitées.

        Arguments :
            data (pd.DataFrame) : Les données prétraitées.
        """
        if data.empty:
            raise ValueError("Data is empty. Please provide a valid DataFrame.")
        self.data = data

    def apply_dbscan(self, eps=5, min_samples=5):
        """
        Applique DBSCAN après normalisation des données et traitement des outliers.

        Arguments :
            eps (float) : Distance maximale entre deux échantillons pour qu'ils soient considérés comme voisins.
            min_samples (int) : Nombre minimum d'échantillons dans un voisinage pour qu'un point soit considéré comme un noyau.

        Retour :
            pd.DataFrame : Le DataFrame avec les clusters ajoutés.
        """
        # Standardisation des données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        # Appliquer t-SNE pour réduire à 2D
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(scaled_data)

        # Appliquer DBSCAN sur les résultats de t-SNE
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        clusters = dbscan.fit_predict(X_tsne)

        # Ajouter les clusters au DataFrame
        self.data["tsne1"], self.data["tsne2"] = X_tsne[:, 0], X_tsne[:, 1]
        self.data["cluster"] = clusters

        return self.data

    def get_kdist_plot(self, k):
        """
        Affiche le graphique des distances k-distances pour déterminer le bon paramètre eps pour DBSCAN.

        Arguments :
            k (int) : Nombre de voisins à utiliser pour le calcul des distances.
        """
        # Standardisation des données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        nbrs = NearestNeighbors(n_neighbors=k).fit(scaled_data)
        distances, _ = nbrs.kneighbors(scaled_data)

        distances = np.sort(distances, axis=0)
        distances = distances[:, k-1]

        plt.figure(figsize=(8, 8))
        plt.plot(distances)
        plt.xlabel('Points/Objects in the dataset')
        plt.ylabel(f'Sorted {k}-nearest neighbor distance')
        plt.grid(True, linestyle="--", color='black', alpha=0.4)
        plt.show()


class Plotter:
    """
    Classe pour visualiser les résultats du clustering.
    """

    @staticmethod
    def plot_clusters(data):
        """
        Visualise les clusters DBSCAN après t-SNE, en mettant en évidence les points de bruit.

        Arguments :
            data (pd.DataFrame) : Le DataFrame contenant les données et les clusters.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot clusters
        sns.scatterplot(
            x=data["tsne1"], y=data["tsne2"], hue=data["cluster"], palette="tab10", alpha=0.5, edgecolor='w'
        )
        
        # Highlight noise points
        noise = data[data["cluster"] == -1]
        plt.scatter(noise["tsne1"], noise["tsne2"], color='red', label='Noise', alpha=0.5, edgecolor='w')
        
        plt.title("Clusters DBSCAN après t-SNE")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.show()
        # Ajouter des labels et un titre
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("Nuage de points des données après t-SNE et K-Means")
        plt.show()
