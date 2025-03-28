import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectPercentile, GenericUnivariateSelect, SelectFromModel, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from statsmodels.tools import add_constant
import statsmodels.api as sm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CorrelationFeatureSelector:
    def __init__(self, data):
        """
        Initializes the CorrelationFeatureSelector class.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing the dataset on which feature selection
        based on variance will be performed.

        Attributes:
        data (pd.DataFrame): Stores the input DataFrame for further processing.
        """

        self.data = data

    def select_highly_correlated_features(self, threshold=0.9):
        """
        Select and remove highly correlated features from the DataFrame.

        This method identifies features in the DataFrame that are highly correlated
        with each other based on the specified correlation threshold. It removes one
        of the features from each pair of highly correlated features to reduce redundancy
        and improve model performance.

        Args:
            threshold (float, optional): The correlation threshold above which features
            are considered highly correlated. Defaults to 0.9.

        Returns:
            pandas.DataFrame: The DataFrame with highly correlated features removed.

        Side Effects:
            - Updates the `self.data` attribute by dropping the identified features.
            - Logs the names of the removed features and the new shape of the DataFrame.

        Example:
            >>> feature_engineer = CorrelationFeatureSelector(data)
            >>> data_cleaned = feature_engineer.select_highly_correlated_features(threshold=0.85)
        """
        corr_matrix = self.data.corr(numeric_only=True)
        features_to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    features_to_drop.add(colname)

        self.data = self.data.drop(columns=features_to_drop)
        logging.info(f"Features removed: {features_to_drop}")
        logging.info(f"New shape of the DataFrame: {self.data.shape}")

        # Save the data
        saver = SaveData(self.data)
        saver.save_data()
        
        
        return self.data

    def run(self, threshold=0.9):
        """
        Exécute la sélection des features hautement corrélées.

        Arguments :
        threshold (float) : Le seuil de corrélation au-delà duquel les features seront supprimées.

        Retour :
        pd.DataFrame : Le DataFrame après suppression des features corrélées.
        """
        return self.select_highly_correlated_features(threshold=threshold)

class VarianceFeatureSelector:
    def __init__(self, data):
        """
        Initializes the VarianceFeatureSelector class.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing the dataset on which feature selection
        based on variance will be performed.

        Attributes:
        data (pd.DataFrame): Stores the input DataFrame for further processing.
        """
        self.data = data

    def select_features_by_variance(self, threshold=0.8):
        """
        Selects features from the DataFrame with variance above a specified threshold.

        Parameters
        ----------
        threshold : float, optional
            The variance threshold for feature selection. Features with variance below
            this threshold will be removed. Default is 0.8.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing only the selected features with variance above the threshold.

        Notes
        -----
        - Only numerical columns (int64 and float64) are considered for variance calculation.
        - Logs the selected features and the new shape of the DataFrame.
        """
        X = self.data.select_dtypes(include=["int64", "float64"])
        selector = VarianceThreshold(threshold=(threshold * (1 - threshold)))
        selector.fit_transform(X)
        selected_columns = X.columns[selector.get_support()]
        X_selected_data = X[selected_columns]
        logging.info(f"Features sélectionnées : {selected_columns.tolist()}")
        logging.info(f"Nouvelle forme du DataFrame : {X_selected_data.shape}")

        # Save the data
        saver = SaveData(self.data)
        saver.save_data()

        return X_selected_data
    
    def run(self, threshold=0.8):
        """
        Exécute la sélection des features basées sur la variance.

        Arguments :
        threshold (float) : Le seuil de variance en dessous duquel les features seront supprimées.

        Retour :
        pd.DataFrame : Le DataFrame après suppression des features à faible variance.
        """
        return self.select_features_by_variance(threshold=threshold)

class AnovaFeatureSelector:
    def __init__(self, data):
        """
        Initializes the AnovaFeatureSelector class.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing the dataset on which feature selection
        based on ANOVA will be performed.

        Attributes:
        data (pd.DataFrame): Stores the input DataFrame for further processing.
        """
        self.data = data

    def select_anova_k_best_features(self, target_column, score_func, k=10):
        """
        Selects the top k features based on the ANOVA F-value.

        Parameters:
        target_column (str): The name of the target column in the DataFrame.
        k (int): The number of top features to select. Default is 10.

        Returns:
        pd.DataFrame: A DataFrame containing only the selected features.
        """
        X = self.data.select_dtypes(include=["int64", "float64"])
        y = self.data[target_column]
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        X_selected_data = X[selected_columns]
        logging.info(f"Features sélectionnées : {selected_columns.tolist()}")
        logging.info(f"Nouvelle forme du DataFrame : {X_selected_data.shape}")
        
        # Save the data
        saver = SaveData(X_selected_data)
        saver.save_data()   

        return X_selected_data

    def select_percentile_best_features(self, target_column, score_func, percentile=10):
        """
        Selects the top features based on a specified percentile of the ANOVA F-value.

        Parameters:
        target_column (str): The name of the target column in the DataFrame.
        percentile (int): The percentile of top features to select. Default is 10.

        Returns:
        pd.DataFrame: A DataFrame containing only the selected features.
        """
        X = self.data.select_dtypes(include=["int64", "float64"])
        y = self.data[target_column]
        selector = SelectPercentile(score_func=score_func, percentile=percentile)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        X_selected_data = X[selected_columns]
        logging.info(f"Features sélectionnées : {selected_columns.tolist()}")
        logging.info(f"Nouvelle forme du DataFrame : {X_selected_data.shape}")

        # Save the data
        saver = SaveData(X_selected_data)
        saver.save_data() 

        return X_selected_data

    def run(self, target_column, method="anova_k_best", param=10):
        """
        Executes the feature selection process based on the specified method.

        Parameters:
        target_column (str): The name of the target column in the DataFrame.
        method (str): The method to use for feature selection ('anova_k_best' or 'percentile'). Default is 'anova_k_best'.
        param (int): The parameter for the selected method (k for 'anova_k_best' or percentile for 'percentile'). Default is 10.

        Returns:
        pd.DataFrame: A DataFrame containing the selected features.
        """
        if method == "anova_k_best":
            return self.select_anova_k_best_features(target_column, k=param)
        elif method == "percentile":
            return self.select_percentile_best_features(target_column, percentile=param)
        else:
            raise ValueError("Invalid method. Choose 'anova_k_best' or 'percentile'.")

class GenericUnivariateFeatureSelector:
    """
    Classe pour effectuer une sélection univariée générique des features en utilisant différentes méthodes.

    Attributes:
        data (pd.DataFrame): Le DataFrame contenant les données sur lesquelles la sélection sera effectuée.
    """

    def __init__(self, data):
        """
        Initialise la classe avec le DataFrame.

        Arguments :
            data (pd.DataFrame) : Les données à analyser.
        """
        self.data = data

    def select_generic_univariate_features(self, target_column, score_func, mode='percentile', param=10):
        """
        Sélectionne les features en utilisant une méthode univariée générique.

        Arguments :
            target_column (str) : Le nom de la colonne cible dans le DataFrame.
            mode (str) : La méthode de sélection ('percentile', 'k_best', 'fpr', 'fdr', 'fwe'). Par défaut 'percentile'.
            param (int ou float) : Le paramètre pour la méthode sélectionnée (ex: percentile ou k). Par défaut 10.

        Retour :
            pd.DataFrame : Un DataFrame contenant uniquement les features sélectionnées.
        """
        X = self.data.select_dtypes(include=["int64", "float64"])
        y = self.data[target_column]
        selector = GenericUnivariateSelect(score_func=score_func, mode=mode, param=param)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        X_selected_data = X[selected_columns]
        logging.info(f"Features sélectionnées : {selected_columns.tolist()}")
        logging.info(f"Nouvelle forme du DataFrame : {X_selected_data.shape}")

        # Save the data
        saver = SaveData(X_selected_data)
        saver.save_data() 

        return X_selected_data

    def run(self, target_column, mode='percentile', param=10):
        """
        Exécute la sélection univariée générique des features.

        Arguments :
            target_column (str) : Le nom de la colonne cible dans le DataFrame.
            mode (str) : La méthode de sélection ('percentile', 'k_best', 'fpr', 'fdr', 'fwe'). Par défaut 'percentile'.
            param (int ou float) : Le paramètre pour la méthode sélectionnée (ex: percentile ou k). Par défaut 10.

        Retour :
            pd.DataFrame : Un DataFrame contenant uniquement les features sélectionnées.
        """
        return self.select_generic_univariate_features(target_column, mode=mode, param=param)

class ModelFeatureSelector:
    """
    Classe pour effectuer une sélection de features basée sur des modèles d'apprentissage automatique.

    Attributes:
        data (pd.DataFrame): Le DataFrame contenant les données sur lesquelles la sélection sera effectuée.
    """

    def __init__(self, data):
        """
        Initialise la classe avec le DataFrame.

        Arguments :
            data (pd.DataFrame) : Les données à analyser.
        """
        self.data = data

    def select_random_forest_features(self, target_column, threshold_model='mean', max_features=None):
        """
        Sélectionne les features en utilisant un modèle RandomForestClassifier.

        Arguments :
            target_column (str) : Le nom de la colonne cible dans le DataFrame.
            threshold_model (str ou float) : Le seuil pour la sélection des features. Par défaut 'mean'.
            max_features (int ou None) : Le nombre maximum de features à sélectionner. Par défaut None.

        Retour :
            pd.DataFrame : Un DataFrame contenant uniquement les features sélectionnées.
        """
        X = self.data.select_dtypes(include=["int64", "float64"])
        y = self.data[target_column]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        selector = SelectFromModel(estimator=model, threshold=threshold_model, max_features=max_features, prefit=True)
        selected_columns = X.columns[selector.get_support()]
        X_selected_data = X[selected_columns]
        logging.info(f"Features sélectionnées : {selected_columns.tolist()}")
        logging.info(f"Nouvelle forme du DataFrame : {X_selected_data.shape}")

        # Save the data
        saver = SaveData(X_selected_data)
        saver.save_data() 

        return X_selected_data

    def select_lasso_features(self, target_column, alpha=1.0, threshold_model='mean', max_features=None):
        """
        Sélectionne les features en utilisant un modèle Lasso.

        Arguments :
            target_column (str) : Le nom de la colonne cible dans le DataFrame.
            alpha (float) : Le paramètre de régularisation pour le modèle Lasso. Par défaut 1.0.
            threshold_model (str ou float) : Le seuil pour la sélection des features. Par défaut 'mean'.
            max_features (int ou None) : Le nombre maximum de features à sélectionner. Par défaut None.

        Retour :
            pd.DataFrame : Un DataFrame contenant uniquement les features sélectionnées.
        """
        X = self.data.select_dtypes(include=["int64", "float64"])
        y = self.data[target_column]
        model = Lasso(alpha=alpha, random_state=42)
        model.fit(X, y)
        selector = SelectFromModel(estimator=model, threshold=threshold_model, max_features=max_features, prefit=True)
        selected_columns = X.columns[selector.get_support()]
        X_selected_data = X[selected_columns]
        logging.info(f"Features sélectionnées : {selected_columns.tolist()}")
        logging.info(f"Nouvelle forme du DataFrame : {X_selected_data.shape}")

        # Save the data
        saver = SaveData(X_selected_data)
        saver.save_data() 

        return X_selected_data

    def run(self, target_column, method="random_forest", **kwargs):
        """
        Exécute la sélection des features basée sur le modèle spécifié.

        Arguments :
            target_column (str) : Le nom de la colonne cible dans le DataFrame.
            method (str) : La méthode de sélection ('random_forest' ou 'lasso'). Par défaut 'random_forest'.
            **kwargs : Arguments supplémentaires pour les méthodes de sélection.

        Retour :
            pd.DataFrame : Un DataFrame contenant les features sélectionnées.
        """
        if method == "random_forest":
            return self.select_random_forest_features(target_column, **kwargs)
        elif method == "lasso":
            return self.select_lasso_features(target_column, **kwargs)
        else:
            raise ValueError("Invalid method. Choose 'random_forest' or 'lasso'.")

class SequentialFeatureSelector:
    """
    Classe pour effectuer une sélection séquentielle des features en utilisant des modèles de régression.

    Attributes:
        data (pd.DataFrame): Le DataFrame contenant les données sur lesquelles la sélection sera effectuée.
    """

    def __init__(self, data):
        """
        Initialise la classe avec le DataFrame.

        Arguments :
            data (pd.DataFrame) : Les données à analyser.
        """
        self.data = data

    def select_sequential_features(self, target_column, n_features_to_select=10, direction='forward'):
        """
        Sélectionne les features en utilisant une sélection séquentielle basée sur un modèle de régression logistique.

        Arguments :
            target_column (str) : Le nom de la colonne cible dans le DataFrame.
            n_features_to_select (int) : Le nombre de features à sélectionner. Par défaut 10.
            direction (str) : La direction de la sélection ('forward' ou 'backward'). Par défaut 'forward'.

        Retour :
            pd.DataFrame : Un DataFrame contenant uniquement les features sélectionnées.
        """
        X = self.data.select_dtypes(include=["int64", "float64"])
        y = self.data[target_column]
        model = LogisticRegression(max_iter=1000, random_state=42)
        sfs = SequentialFeatureSelector(self.data)
        #sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction=direction)
        sfs.fit(X, y)
        selected_columns = X.columns[sfs.get_support()]
        X_selected_data = X[selected_columns]
        logging.info(f"Features sélectionnées : {selected_columns.tolist()}")
        logging.info(f"Nouvelle forme du DataFrame : {X_selected_data.shape}")

        # Save the data
        saver = SaveData(X_selected_data)
        saver.save_data() 

        return X_selected_data

    def select_sequential_features_aic(self, target_column, n_features_to_select=10, direction='forward'):
        """
        Sélectionne les features en utilisant une sélection séquentielle basée sur le critère AIC.

        Arguments :
            target_column (str) : Le nom de la colonne cible dans le DataFrame.
            n_features_to_select (int) : Le nombre de features à sélectionner. Par défaut 10.
            direction (str) : La direction de la sélection ('forward' ou 'backward'). Par défaut 'forward'.

        Retour :
            pd.DataFrame : Un DataFrame contenant uniquement les features sélectionnées.
        """
        X = self.data.select_dtypes(include=["int64", "float64"])
        y = self.data[target_column]
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
            logging.info(f"Selected feature: {best_feature}, AIC: {best_aic}")

        X_selected_data = X[selected_features]
        logging.info(f"Features sélectionnées : {selected_features}")
        logging.info(f"Nouvelle forme du DataFrame : {X_selected_data.shape}")

        # Save the data
        saver = SaveData(X_selected_data)
        saver.save_data() 

        return X_selected_data

    def run(self, target_column, method="sequential", **kwargs):
        """
        Exécute la sélection des features en fonction de la méthode spécifiée.

        Arguments :
            target_column (str) : Le nom de la colonne cible dans le DataFrame.
            method (str) : La méthode de sélection ('sequential' ou 'aic'). Par défaut 'sequential'.
            **kwargs : Arguments supplémentaires pour les méthodes de sélection.

        Retour :
            pd.DataFrame : Un DataFrame contenant les features sélectionnées.
        """
        if method == "sequential":
            return self.select_sequential_features(target_column, **kwargs)
        elif method == "aic":
            return self.select_sequential_features_aic(target_column, **kwargs)
        else:
            raise ValueError("Invalid method. Choose 'sequential' or 'aic'.")
        
class SaveData:
    def __init__(self, data):
        """
        Initializes the SaveData class with the dataset.

        :param data: The DataFrame to be saved.
        """
        self.data = data

    def save_data(self, save_path="./data/processed/processed.csv"):
        """
        Save the processed data to a CSV file and display its information.

        :param save_path: The path where the file will be saved. Default is './data/processed/processed_sample_10000.csv'.
        """
        self.data.to_csv(save_path, index=False)
        logging.info(f"Processed data saved to '{save_path}'.")
        self.data.info()
    

class FeatureEngineering:
    """
    Factory class to create and execute feature selectors based on the provided method.
    """

    def __init__(self, file_path):
        """
        Initializes the FeatureEngineering class with the dataset path.
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        logging.info("Sample Dataset loaded successfully!")

    def feature_selection(self, method, **kwargs):
        """
        Creates and executes the appropriate feature selector based on the provided method.

        Arguments:
            method (str): The feature selection method to use.
            **kwargs: Additional arguments for the specific feature selection method.

        Returns:
            pd.DataFrame: The DataFrame with selected features.
        """
        logging.info(f"Starting feature selection using method: {method}")

        if method == "correlation":
            selector = CorrelationFeatureSelector(self.data)
            # Remove useless keys from kwargs for correlation method
            keys_to_pop = [
                "target_column", "k", "percentile", "mode", "param", "score_func", "threshold_model", 
                "n_features_to_select", "direction"
            ]
            for key in keys_to_pop:
                kwargs.pop(key, None)
            return selector.select_highly_correlated_features(**kwargs)

        elif method == "variance":
            selector = VarianceFeatureSelector(self.data)
            # Remove useless keys from kwargs for variance method
            keys_to_pop = [
                "target_column", "k", "percentile", "mode", "param", "score_func", "threshold_model", 
                "n_features_to_select", "direction"
            ]
            for key in keys_to_pop:
                kwargs.pop(key, None)
            return selector.select_features_by_variance(**kwargs)

        elif method in ["anova_k_best", "anova_percentile"]:
            if not kwargs["target_column"]:
                raise ValueError("Target column is required for ANOVA-based methods.")
            selector = AnovaFeatureSelector(self.data)
            if method == "anova_k_best":
                # Remove useless keys from kwargs for variance method
                keys_to_pop = [
                    "threshold", "percentile", "mode", "param", "threshold_model", 
                    "n_features_to_select", "direction"
                ]
                for key in keys_to_pop:
                    kwargs.pop(key, None)
                return selector.select_anova_k_best_features(target_column=kwargs.pop("target_column"), **kwargs)
            else:
                # Remove useless keys from kwargs for variance method
                keys_to_pop = [
                    "threshold", "k", "mode", "param", "threshold_model", 
                    "n_features_to_select", "direction"
                ]
                for key in keys_to_pop:
                    kwargs.pop(key, None)
                return selector.select_percentile_best_features(target_column=kwargs.pop("target_column"), **kwargs)

        elif method == "generic_univariate":
            if not kwargs["target_column"]:
                raise ValueError("Target column is required for univariate methods.")
            selector = GenericUnivariateFeatureSelector(self.data)
            # Remove useless keys from kwargs for variance method
            keys_to_pop = [
                "threshold", "k", "percentile", "threshold_model", 
                "n_features_to_select", "direction"
            ]
            for key in keys_to_pop:
                kwargs.pop(key, None)
            return selector.select_generic_univariate_features(target_column=kwargs.pop("target_column"), **kwargs)

        elif method in ["random_forest", "lasso"]:
            if not kwargs["target_column"]:
                raise ValueError("Target column is required for model-based methods.")
            selector = ModelFeatureSelector(self.data)
            if method == "random_forest":
                # # Remove useless keys from kwargs for variance method
                # keys_to_pop = [
                #     "threshold", "k", "percentile", "mode", "param", "score_func", 
                #     "n_features_to_select", "direction"
                # ]
                # for key in keys_to_pop:
                #     kwargs.pop(key, None)
                return selector.select_random_forest_features(target_column=kwargs.pop("target_column"), **kwargs)
            else:
                # Remove useless keys from kwargs for variance method
                keys_to_pop = [
                    "threshold", "k", "percentile", "mode", "param", "score_func", 
                    "n_features_to_select", "direction"
                ]
                for key in keys_to_pop:
                    kwargs.pop(key, None)
                return selector.select_lasso_features(target_column=kwargs.pop("target_column"), **kwargs)

        elif method in ["sequential", "aic"]:
            if not kwargs["target_column"]:
                raise ValueError("Target column is required for sequential methods.")
            selector = SequentialFeatureSelector(self.data)
            if method == "sequential":
                # Remove useless keys from kwargs for variance method
                keys_to_pop = [
                    "threshold", "k", "percentile", "mode", "param", "score_func", 
                    "threshold_model"
                ]
                for key in keys_to_pop:
                    kwargs.pop(key, None)
                return selector.select_sequential_features(target_column=kwargs.pop("target_column"), **kwargs)
            else:
                # Remove useless keys from kwargs for variance method
                keys_to_pop = [
                    "threshold", "k", "percentile", "mode", "param", "score_func", 
                    "threshold_model"
                ]
                for key in keys_to_pop:
                    kwargs.pop(key, None)
                return selector.select_sequential_features_aic(target_column=kwargs.pop("target_column"), **kwargs)

        else:
            logging.error(f"Unknown method: {method}")
            raise ValueError(f"Unknown method: {method}")

    def run(self, method, **kwargs):
        """
        Executes the feature selection process using the specified method.

        Arguments:
            method (str): The feature selection method to use.
            **kwargs: Additional arguments for the specific feature selection method.

        Returns:
            pd.DataFrame: The DataFrame with selected features.
        """
        return self.feature_selection(method, **kwargs)