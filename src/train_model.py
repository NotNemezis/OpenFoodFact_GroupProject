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

    def remove_irrelevant_columns(self):
        """
        Supprime les colonnes non pertinentes pour l'analyse.

        Retour :
            pd.DataFrame : Le DataFrame nettoyé, sans les colonnes jugées inutiles.
        """
        columns_to_drop = [
            "code", "url", "creator", "created_t", "created_datetime",
            "last_modified_t", "last_modified_datetime", "packaging", "packaging_tags",
            "brands_tags", "categories_tags", "categories_fr",
            "origins_tags", "manufacturing_places", "manufacturing_places_tags",
            "labels_tags", "labels_fr", "emb_codes", "emb_codes_tags",
            "first_packaging_code_geo", "cities", "cities_tags", "purchase_places",
            "countries_tags", "countries_fr", "image_ingredients_url",
            "image_ingredients_small_url", "image_nutrition_url", "image_nutrition_small_url",
            "image_small_url", "image_url", "last_updated_t", "last_updated_datetime",
            "last_modified_by", "last_image_t", "last_image_datetime"
        ]

        self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns], errors='ignore', inplace=True)
        return self.df

    def remove_duplicates(self):
        """
        Supprime les doublons dans le DataFrame.

        Retour :
            pd.DataFrame : Le DataFrame sans doublons.
        """
        self.df.drop_duplicates(keep="first", inplace=True)
        return self.df

    def remove_high_nan_columns(self, threshold=70):
        """
        Supprime les colonnes contenant un pourcentage de valeurs manquantes supérieur au seuil défini.

        Arguments :
            threshold (float) : Pourcentage maximal de valeurs manquantes toléré dans une colonne.
                                Par défaut, ce seuil est fixé à 70%.

        Retour :
            pd.DataFrame : Le DataFrame après suppression des colonnes trop incomplètes.
        """
        nan_ratio = self.df.isna().mean() * 100  # Calcul du pourcentage de valeurs NaN par colonne
        cols_to_remove = nan_ratio[nan_ratio > threshold].index.tolist()  # Sélection des colonnes à supprimer

        self.df.drop(columns=cols_to_remove, inplace=True)  # Suppression des colonnes sélectionnées
        return self.df

    def pre_processing(self):
        """
        Effectue les étapes de prétraitement suivantes sur le DataFrame :
        1. Sélectionne les colonnes avec '_100g' dans leur nom.
        2. Supprime les colonnes fortement corrélées (seuil > 0.8).
        3. Supprime les lignes avec des valeurs manquantes.
        4. Standardise les données (normalisation).

        Retour :
            pd.DataFrame : Les données normalisées et traitées.
        """
        if self.df.empty:
            print("DataFrame is empty. Please check the file path and try again.")
            return self.df

        # Sélectionner les colonnes '_100g'
        col100g = [col for col in self.df.columns if "_100g" in col]
        data_100g = self.df[col100g]

        # Calcul de la matrice de corrélation
        corr_matrix = data_100g.corr().abs()

        # Déterminer les colonnes fortement corrélées (seuil > 0.8)
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

        # Supprimer les colonnes redondantes
        data_100g = data_100g.drop(columns=to_drop)

        # Supprimer les lignes contenant des valeurs manquantes
        data_100g = data_100g.dropna()

        return data_100g


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

        # Ajouter des labels et un titre
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("Nuage de points des données après t-SNE et K-Means")
        plt.show()
