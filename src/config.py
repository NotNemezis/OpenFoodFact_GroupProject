# # Path configurations
DATA_PATH = "data/dataset/en.openfoodfacts.org.products.csv"
SAMPLE_PATH = "data/dataset/sample_10000.csv"
# MODEL_SAVE_PATH = "data/results/trained_model.pkl"
# RESULTS_PATH = "data/results/model_results.json"

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
