import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE

class PCAAnalyzer:
    """
    Classe pour l'Analyse en Composantes Principales (ACP) sur les données prétraitées.
    """

    def __init__(self, data):
        """
        Initialise la classe avec les données prétraitées.

        Arguments :
            data (pd.DataFrame) : Les données prétraitées.
        """
        self.data = data
        self.scaled_data = None
        self.pca_components = None
        self.explained_variance_ratio = None

    def apply_pca(self, n_components=5):
        """
        Applique la standardisation des données et effectue une Analyse en Composantes Principales (ACP).

        Arguments :
            n_components (int) : Nombre de composantes principales à conserver (par défaut 5).

        Retour :
            tuple :
                - np.ndarray : Les données projetées dans l'espace des composantes principales.
                - np.ndarray : La proportion de variance expliquée par chaque composante.
        """
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)

        # Application de l'ACP
        pca = PCA(n_components=n_components)
        self.pca_components = pca.fit_transform(self.scaled_data)

        # Récupération de la variance expliquée par chaque composante principale
        self.explained_variance_ratio = pca.explained_variance_ratio_

        return self.pca_components, self.explained_variance_ratio

    def plot_explained_variance(self):
        """
        Génère un graphique illustrant la variance expliquée par chaque composante principale.
        """
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, len(self.explained_variance_ratio) + 1), self.explained_variance_ratio, alpha=0.7)
        plt.xlabel('Composantes principales')
        plt.ylabel('Variance expliquée')
        plt.title('Variance expliquée par chaque composante principale')
        plt.show()

    def cumulative_variance_plot(self):
        """
        Affiche un graphique de la variance expliquée cumulée.
        """
        cumulative_variance = np.cumsum(self.explained_variance_ratio)

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b', label='Variance cumulée')
        plt.xlabel('Nombre de composantes principales')
        plt.ylabel('Variance cumulée expliquée')
        plt.title('Variance cumulée expliquée')

        plt.axhline(y=0.95, color='r', linestyle='-', label='95% Variance expliquée')
        plt.axvline(x=np.argmax(cumulative_variance >= 0.95) + 1, color='r', linestyle='--', label=f'{np.argmax(cumulative_variance >= 0.95) + 1} composantes')

        plt.legend()
        plt.show()

    def plot_pca_biplot(self, truncate_outliers=True, threshold=1.5):
        """
        Affiche un biplot de l'ACP combinant :
        - La projection des observations sur les deux premières composantes principales.
        - Les vecteurs des variables contribuant à ces composantes.
        """
        data_100g = self.data

        if truncate_outliers:
            Q1 = data_100g.quantile(0.25)
            Q3 = data_100g.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            data_100g = data_100g[~((data_100g < lower_bound) | (data_100g > upper_bound)).any(axis=1)]

        scaled_data_100g = StandardScaler().fit_transform(data_100g)

        pca = PCA(n_components=2)
        pca_scores = pca.fit_transform(scaled_data_100g)
        loadings = pca.components_.T

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=pca_scores[:, 0], y=pca_scores[:, 1], alpha=0.5, label="Observations")

        for i, var in enumerate(data_100g.columns):
            plt.arrow(0, 0, loadings[i, 0] * 3, loadings[i, 1] * 3, color='r', alpha=0.5, head_width=0.05)
            plt.text(loadings[i, 0] * 3.2, loadings[i, 1] * 3.2, var, color='r', fontsize=10)

        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.title("Biplot de l'ACP")
        plt.legend()
        plt.show()

    def plot_pca_scatter(self, truncate_outliers=True, threshold=1.5):
        """
        Affiche un nuage de points des données projetées sur les deux premières composantes principales.
        """
        if self.pca_components is None:
            raise ValueError("L'ACP n'a pas encore été appliquée. Exécutez `apply_pca()` d'abord.")

        data_100g = self.data

        if truncate_outliers:
            Q1 = data_100g.quantile(0.25)
            Q3 = data_100g.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            data_100g = data_100g[~((data_100g < lower_bound) | (data_100g > upper_bound)).any(axis=1)]

        scaled_data_100g = StandardScaler().fit_transform(data_100g)

        pca = PCA(n_components=2)
        pca_scores = pca.fit_transform(scaled_data_100g)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=pca_scores[:, 0], y=pca_scores[:, 1], alpha=0.5)
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.title("Nuage de points des données après ACP")
        plt.show()


class OPTICSAnalyzer:
    """
    Classe pour appliquer OPTICS sur les données prétraitées.
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

    def apply_optics(self, min_samples=40, xi=0.02, min_cluster_size=0.1):
        """
        Applique OPTICS après normalisation des données et traitement des outliers.

        Arguments :
            min_samples (int) : Nombre minimum d'échantillons dans un voisinage pour qu'un point soit considéré comme un noyau.
            xi (float) : Seuil de pente pour la formation des clusters.
            min_cluster_size (float) : Taille minimale d'un cluster en pourcentage du nombre total de points.

        Retour :
            pd.DataFrame : Le DataFrame avec les clusters ajoutés.
        """
        # Standardisation des données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        # Appliquer t-SNE pour réduire à 2D
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(scaled_data)

        # Appliquer OPTICS sur les résultats de t-SNE
        optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        clusters = optics.fit_predict(X_tsne)

        # Ajouter les clusters au DataFrame
        self.data["tsne1"], self.data["tsne2"] = X_tsne[:, 0], X_tsne[:, 1]
        self.data["cluster"] = clusters

        return self.data

    def get_reachability_plot(self):
        """
        Affiche le graphique de la distance de portée pour visualiser la structure des clusters.

        """
        # Standardisation des données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        # Appliquer OPTICS
        optics = OPTICS(min_samples=40, xi=0.02, min_cluster_size=0.1)
        optics.fit(scaled_data)

        reachability = optics.reachability_[optics.ordering_]
        labels = optics.labels_[optics.ordering_]

        plt.figure(figsize=(10, 6))
        plt.plot(reachability)
        plt.xlabel('Points/Objects in the dataset')
        plt.ylabel('Reachability Distance')
        plt.title('Reachability Plot')
        plt.grid(True, linestyle="--", color='black', alpha=0.4)
        plt.show()

class Plotter:
    """
    Classe pour visualiser les résultats du clustering.
    """

    @staticmethod
    def plot_clusters(data):
        """
        Visualise les clusters OPTICS après t-SNE, en mettant en évidence les points de bruit.

        Arguments :
            data (pd.DataFrame) : Le DataFrame contenant les données et les clusters.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot clusters
        sns.scatterplot(
            x=data["tsne1"], y=data["tsne2"], hue=data["cluster"], palette="viridis", alpha=0.5, edgecolor='w'
        )
        
        # Highlight noise points
        noise = data[data["cluster"] == -1]
        plt.scatter(noise["tsne1"], noise["tsne2"], color='red', label='Noise', alpha=0.5, edgecolor='w')
        
        plt.title("Clusters OPTICS après t-SNE")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.show()

