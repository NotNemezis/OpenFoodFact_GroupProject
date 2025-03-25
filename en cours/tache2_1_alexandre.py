import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, VarianceThreshold, f_classif, SelectPercentile,
    GenericUnivariateSelect, SelectFromModel, SequentialFeatureSelector
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from statsmodels.tools import add_constant
import statsmodels.api as sm


class Tache21:
    def __init__(self, file_path="datasets/en.openfoodfacts.org.products.csv", nrows=100000):
        """
        Initialise la classe en chargeant un échantillon du dataset Open Food Facts.

        Arguments :
            file_path (str) : Chemin du fichier CSV à charger.
            nrows (int) : Nombre de lignes à charger (par défaut 100 000).
        """
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        self.df = pd.read_csv(
            file_path,
            sep="\t",
            on_bad_lines='skip',
            nrows=nrows,
            low_memory=False
        )
        print(self.df.shape)

    def select_highly_correlated_features(self, threshold=0.9):
        """
        Sélectionne les features les plus corrélées entre elles et les élimine si leur corrélation dépasse un certain seuil.

        Arguments :
            threshold (float) : Seuil de corrélation au-delà duquel les features seront éliminées.

        Retour :
            pd.DataFrame : Un DataFrame avec les features sélectionnées.
        """
        corr_matrix = self.df.corr()
        features_to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    features_to_drop.add(colname)

        self.df = self.df.drop(columns=features_to_drop)
        print(f"Features supprimées: {features_to_drop}")
        print(f"Nouvelle forme du DataFrame: {self.df.shape}")
        return self.df

    def select_features_by_variance(self, threshold=0.8):
        """
        Sélectionne les features dont la variance est supérieure au seuil donné.

        Arguments :
            threshold (float) : Seuil de variance. Les features dont la variance est inférieure à ce seuil seront supprimées.

        Retour :
            pd.DataFrame : DataFrame avec les features sélectionnées.
        """
        X = self.df.select_dtypes(include=["int64", "float64"])
        selector = VarianceThreshold(threshold=(threshold * (1 - threshold)))
        selector.fit_transform(X)
        selected_columns = X.columns[selector.get_support()]
        X_selected_df = X[selected_columns]
        print(f"Features sélectionnées : {selected_columns.tolist()}")
        print(f"Nouvelle forme du DataFrame : {X_selected_df.shape}")
        return X_selected_df

    def select_k_best_features(self, target_column, k=10):
        """
        Sélectionne les K meilleures features basées sur la valeur F ANOVA.

        Arguments :
            target_column (str) : Nom de la colonne cible.
            k (int) : Nombre de features à sélectionner.

        Retour :
            pd.DataFrame : DataFrame avec les features sélectionnées.
        """
        X = self.df.select_dtypes(include=["int64", "float64"])
        y = self.df[target_column]
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        X_selected_df = X[selected_columns]
        print(f"Features sélectionnées : {selected_columns.tolist()}")
        print(f"Nouvelle forme du DataFrame : {X_selected_df.shape}")
        return X_selected_df

    def select_percentile_best_features(self, target_column, percentile=10):
        """
        Sélectionne les meilleures features basées sur un certain percentile de la valeur F ANOVA.

        Arguments :
            target_column (str) : Nom de la colonne cible.
            percentile (int) : Percentile de features à sélectionner.

        Retour :
            pd.DataFrame : DataFrame avec les features sélectionnées.
        """
        X = self.df.select_dtypes(include=["int64", "float64"])
        y = self.df[target_column]
        selector = SelectPercentile(score_func=f_classif, percentile=percentile)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        X_selected_df = X[selected_columns]
        print(f"Features sélectionnées : {selected_columns.tolist()}")
        print(f"Nouvelle forme du DataFrame : {X_selected_df.shape}")
        return X_selected_df

    def select_generic_univariate_features(self, target_column, mode='percentile', param=10):
        """
        Sélectionne les features basées sur une méthode univariée générique.

        Arguments :
            target_column (str) : Nom de la colonne cible.
            mode (str) : Mode de sélection ('percentile', 'k_best', 'fpr', 'fdr', 'fwe').
            param (int) : Paramètre pour le mode de sélection (percentile, k, alpha).

        Retour :
            pd.DataFrame : DataFrame avec les features sélectionnées.
        """
        X = self.df.select_dtypes(include=["int64", "float64"])
        y = self.df[target_column]
        selector = GenericUnivariateSelect(score_func=f_classif, mode=mode, param=param)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        X_selected_df = X[selected_columns]
        print(f"Features sélectionnées : {selected_columns.tolist()}")
        print(f"Nouvelle forme du DataFrame : {X_selected_df.shape}")
        return X_selected_df

    def select_from_model_features(self, target_column, threshold='mean', max_features=None):
        """
        Sélectionne les features basées sur l'importance des features d'un modèle.

        Arguments :
            target_column (str) : Nom de la colonne cible.
            threshold (str or float) : Seuil pour l'importance des features ('mean', 'median', float multiples).
            max_features (int) : Nombre maximum de features à sélectionner.

        Retour :
            pd.DataFrame : DataFrame avec les features sélectionnées.
        """
        X = self.df.select_dtypes(include=["int64", "float64"])
        y = self.df[target_column]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        selector = SelectFromModel(estimator=model, threshold=threshold, max_features=max_features, prefit=True)
        selected_columns = X.columns[selector.get_support()]
        X_selected_df = X[selected_columns]
        print(f"Features sélectionnées : {selected_columns.tolist()}")
        print(f"Nouvelle forme du DataFrame : {X_selected_df.shape}")
        return X_selected_df

    def select_lasso_features(self, target_column, alpha=1.0, threshold='mean', max_features=None):
        """
        Sélectionne les features basées sur l'importance des features d'un modèle Lasso.

        Arguments :
            target_column (str) : Nom de la colonne cible.
            alpha (float) : Paramètre de régularisation L1.
            threshold (str or float) : Seuil pour l'importance des features ('mean', 'median', float multiples).
            max_features (int) : Nombre maximum de features à sélectionner.

        Retour :
            pd.DataFrame : DataFrame avec les features sélectionnées.
        """
        X = self.df.select_dtypes(include=["int64", "float64"])
        y = self.df[target_column]
        model = Lasso(alpha=alpha, random_state=42)
        model.fit(X, y)
        selector = SelectFromModel(estimator=model, threshold=threshold, max_features=max_features, prefit=True)
        selected_columns = X.columns[selector.get_support()]
        X_selected_df = X[selected_columns]
        print(f"Features sélectionnées : {selected_columns.tolist()}")
        print(f"Nouvelle forme du DataFrame : {X_selected_df.shape}")
        return X_selected_df

    def select_sequential_features(self, target_column, n_features_to_select=10, direction='forward'):
        """
        Sélectionne les features en utilisant la sélection de features séquentielle (SFS).

        Arguments :
            target_column (str) : Nom de la colonne cible.
            n_features_to_select (int) : Nombre de features à sélectionner.
            direction (str) : Direction de la sélection ('forward' ou 'backward').

        Retour :
            pd.DataFrame : DataFrame avec les features sélectionnées.
        """
        X = self.df.select_dtypes(include=["int64", "float64"])
        y = self.df[target_column]
        model = LogisticRegression(max_iter=1000, random_state=42)
        sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction=direction)
        sfs.fit(X, y)
        selected_columns = X.columns[sfs.get_support()]
        X_selected_df = X[selected_columns]
        print(f"Features sélectionnées : {selected_columns.tolist()}")
        print(f"Nouvelle forme du DataFrame : {X_selected_df.shape}")
        return X_selected_df

    def select_sequential_features_aic(self, target_column, n_features_to_select=10, direction='forward'):
        """
        Sélectionne les features en utilisant la sélection de features séquentielle (SFS) basée sur AIC.

        Arguments :
            target_column (str) : Nom de la colonne cible.
            n_features_to_select (int) : Nombre de features à sélectionner.
            direction (str) : Direction de la sélection ('forward' ou 'backward').

        Retour :
            pd.DataFrame : DataFrame avec les features sélectionnées.
        """
        X = self.df.select_dtypes(include=["int64", "float64"])
        y = self.df[target_column]
        selected_features = []
        remaining_features = list(X.columns)

        while len(selected_features) < n_features_to_select:
            aic_values = []
            for feature in remaining_features:
                features_to_test = selected_features + [feature]
                X_train = add_constant(X[features_to_test])
                model = sm.Logit(y, X_train).fit(disp=0)
                aic_values.append((model.aic, feature))

            aic_values.sort()
            best_aic, best_feature = aic_values[0]
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            print(f"Selected feature: {best_feature}, AIC: {best_aic}")

        X_selected_df = X[selected_features]
        print(f"Features sélectionnées : {selected_features}")
        print(f"Nouvelle forme du DataFrame : {X_selected_df.shape}")
        return X_selected_df


# Example usage of the Tache21 class methods

# Initialize the Tache21 class
tache21 = Tache21()

# Select highly correlated features
print("Selecting highly correlated features...")
df_no_high_corr = tache21.select_highly_correlated_features(threshold=0.9)
print(df_no_high_corr.head())

# Select features by variance
print("\nSelecting features by variance...")
df_high_variance = tache21.select_features_by_variance(threshold=0.8)
print(df_high_variance.head())

# Select K best features
print("\nSelecting K best features...")
df_k_best = tache21.select_k_best_features(target_column='your_target_column', k=10)
print(df_k_best.head())

# Select percentile best features
print("\nSelecting percentile best features...")
df_percentile_best = tache21.select_percentile_best_features(target_column='your_target_column', percentile=10)
print(df_percentile_best.head())

# Select generic univariate features
print("\nSelecting generic univariate features...")
df_generic_univariate = tache21.select_generic_univariate_features(target_column='your_target_column', mode='percentile', param=10)
print(df_generic_univariate.head())

# Select features from model
print("\nSelecting features from model...")
df_from_model = tache21.select_from_model_features(target_column='your_target_column', threshold='mean', max_features=None)
print(df_from_model.head())

# Select Lasso features
print("\nSelecting Lasso features...")
df_lasso = tache21.select_lasso_features(target_column='your_target_column', alpha=1.0, threshold='mean', max_features=None)
print(df_lasso.head())

# Select sequential features
print("\nSelecting sequential features...")
df_sequential = tache21.select_sequential_features(target_column='your_target_column', n_features_to_select=10, direction='forward')
print(df_sequential.head())

# Select sequential features based on AIC
print("\nSelecting sequential features based on AIC...")
df_sequential_aic = tache21.select_sequential_features_aic(target_column='your_target_column', n_features_to_select=10, direction='forward')
print(df_sequential_aic.head())