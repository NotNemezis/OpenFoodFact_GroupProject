# # Path configurations
DATA_PATH = "data/dataset/en.openfoodfacts.org.products.csv"
SAMPLE_PATH = "data/dataset/sample_10000.csv"

# # Data preprocessing parameter
PREPROCESSING_PARAMS = {
    "columns_to_drop": ["code", "url", "creator", "created_t", "created_datetime", "last_modified_t", "last_modified_datetime", "packaging", "packaging_tags", "brands_tags", "categories_tags", "categories_fr", "origins_tags", "manufacturing_places", "manufacturing_places_tags", "labels_tags", "labels_fr", "emb_codes", "emb_codes_tags", "first_packaging_code_geo", "cities", "cities_tags", "purchase_places", "countries_tags", "countries_fr", "image_ingredients_url", "image_ingredients_small_url", "image_nutrition_url", "image_nutrition_small_url", "image_small_url", "image_url", "last_updated_t", "last_updated_datetime", "last_modified_by"],  # List of columns to drop, e.g., ["column1", "column2"]
    "missing_value_threshold": 0.3,  # Threshold for dropping columns with missing values
    "imputation_strategy": "knn",  # Options: "mean", "median", "most_frequent", "constant", "knn"
    "knn_neighbors": 5,  # Number of neighbors for KNN imputation
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
