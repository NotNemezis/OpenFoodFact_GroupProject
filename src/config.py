from sklearn.feature_selection import f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression
"""
Configuration file for data preprocessing, feature engineering, and model training pipeline.

Attributes:
    DATA_PATH (str): Path to the raw dataset file.
    SAMPLE_PATH (str): Path to the sample dataset file.
    DATA_PROCESSED_PATH (str): Path to the preprocessed dataset file.

    PREPROCESSING_PARAMS (dict): Parameters for data preprocessing.
        - columns_to_drop (list): List of column names to drop from the dataset.
        - missing_value_threshold (float): Threshold for dropping columns with missing values.
        - imputation_strategy (str): Strategy for imputing missing values. Options: "mean", "median", "most_frequent", "constant", "knn".
        - knn_neighbors (int): Number of neighbors for KNN imputation (used only if imputation_strategy="knn").

    Additional Parameters:
        - method (str): Feature selection method. Options: "correlation", "variance", "anova_k_best", "anova_percentile", "generic_univariate", "random_forest", "lasso", "sequential", "aic".
        - threshold (float): Correlation threshold for "correlation" method.
        - target_column (list): List of target column names for supervised methods.
        - score_func (callable): Scoring function for "anova_k_best", "anova_percentile", and "generic_univariate".
        - k (int): Number of top features to select for "anova_k_best".
        - percentile (int): Percentage of best features to select for "anova_percentile".
        - mode (str): Mode for "generic_univariate". Options: "percentile", "k_best", "fpr", "fdr", "fwe".
        - param (int or float): Parameter for "generic_univariate" depending on the mode.
        - threshold_model (str): Threshold for "random_forest" or "lasso". Options: "mean", "median", "1.25*mean".
        - n_features_to_select (int): Maximum number of features to select for "sequential" or "aic" methods.
        - direction (str): Direction of feature selection for "sequential" or "aic". Options: "forward", "backward".


Example:
    This configuration file can be used to define parameters for the entire data pipeline, including data_loading, preprocessing, and feature_engineering.
"""

# # Path configurations
DATA_PATH = "data/dataset/en.openfoodfacts.org.products.csv"
SAMPLE_PATH = "data/dataset/sample_10000.csv"
DATA_PROCESSED_PATH = "data/pre_processed/preprocessed_sample_10000.csv"

# # Data preprocessing parameter
PREPROCESSING_PARAMS = {
    "columns_to_drop": ["code", "url", "creator", "created_t", "created_datetime", "last_modified_t", "last_modified_datetime", "packaging", "packaging_tags", "brands_tags", "categories_tags", "categories_fr", "origins_tags", "manufacturing_places", "manufacturing_places_tags", "labels_tags", "labels_fr", "emb_codes", "emb_codes_tags", "first_packaging_code_geo", "cities", "cities_tags", "purchase_places", "countries_tags", "countries_fr", "image_ingredients_url", "image_ingredients_small_url", "image_nutrition_url", "image_nutrition_small_url", "image_small_url", "image_url", "last_updated_t", "last_updated_datetime", "last_modified_by"],  # List of columns to drop, e.g., ["column1", "column2"]
    "missing_value_threshold": 0.3,  # Threshold for dropping columns with missing values
    "imputation_strategy": "knn",  # Options: "mean", "median", "most_frequent", "constant", "knn"
    "knn_neighbors": 5,  # Number of neighbors for KNN imputation
}

# Feature engineering parameters
FEATURE_PARAMS = {
    # Common parameters
    "method": "correlation",     # Feature selection method - Options: "correlation", "variance", "anova_k_best", "anova_percentile", "generic_univariate", "random_forest", "lasso", "sequential", "aic"

    # Parameters only for "correlation" and "variance" methods
    "threshold": 0.9,  # Correlation threshold (used only for the "correlation" method)

    ## Parameters for "anova_k_best", "anova_percentile", "generic_univariate", "random_forest", "lasso", "sequential", "aic" methods
    "target_column": ['energy_100g'],  # ex: ["code", "url"]

    ## Parameters for "anova_k_best", "anova_percentile", "generic_univariate"
    "score_func": f_classif,  # Options: f_classif, f_regression, mutual_info_regression

    ## Parameters for "anova_k_best" method
    "k": 10,  # Number of top features to select

    ## Parameters for "anova_percentile" method
    "percentile": 10,  # Percentage of the best features to select 

    ## Parameters for "generic_univariate" method
    "mode": "fdr",  # Options: "percentile", "k_best", "fpr", "fdr", "fwe"
    ##     - If mode="percentile", param is a percentage (e.g., param=20 for 20% of the best features).
    ##     - If mode="k_best", param is an integer (e.g., param=5 to keep the 5 best features).
    ##     - If mode="fpr", "fdr", or "fwe", param is a p-value threshold (e.g., param=0.05 for a 5% p-value threshold).
    "param": 10, # - param (int or float): Parameter for "generic_univariate" depending on the mode:

    # Parameters for "random_forest" or "lasso" methods
    "threshold_model": "mean",  # Options: "mean", "median", "1.25*mean"

    # Parameters for "sequential" or "aic" methods
    "n_features_to_select": 10,  # Maximum number of features to select 
    "direction": "forward",  # Options:"forward" ou "backward"
}

### Exemple of Config file for all the pipeline
# # config.py

# # Path configurations
# DATA_PATH = "data/raw/products.csv"
# MODEL_SAVE_PATH = "data/results/trained_model.pkl"
# RESULTS_PATH = "data/results/model_results.json"

# # Data preprocessing parameters
# PREPROCESSING_PARAMS = {
#     "handle_missing": "impute_mean",  # Options: "drop", "impute_mean", "impute_median", "fill_custom"
#     "drop_duplicates": True,
#     "encoding": "onehot",  # Options: "label", "onehot"
#     "scaling": "standard",  # Options: "minmax", "standard", "robust"
# }

# # Feature engineering parameters
# FEATURE_PARAMS = {
#     "features_to_keep": ["product_name", "calories", "sugar", "fat", "protein"],
#     "derive_features": True,
#     "log_transform": False,
#     "polynomial_features": False,
# }

# # Train/Test split parameters
# SPLIT_PARAMS = {
#     "train_size": 0.8,
#     "random_state": 42,
#     "stratified": False,
# }

# # Model training parameters
# MODEL_PARAMS = {
#     "model_type": "RandomForest",  # Options: "RandomForest", "XGBoost", "NeuralNetwork"
#     "hyperparameters": {
#         "n_estimators": 100,
#         "max_depth": 10,
#         "random_state": 42,
#     },
# }

# # Model evaluation parameters
# METRIC_PARAMS = {
#     "metrics": ["accuracy", "precision", "recall", "f1_score"],
#     "cross_validation": True,
#     "k_folds": 5,
# }
