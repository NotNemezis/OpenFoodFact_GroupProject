import argparse
import logging
import sys
import os

# Add the parent directory of 'src' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import data_loading, preprocessing, config


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Starting the data science pipeline...")

    # Step 1: Load data
    # logging.info("Loading data...")
    # data_loader = data_loading.DataLoader(config.DATA_PATH)
    # data_loader.run(sample_size=10000, save_path="./data/dataset/sample_10000.csv", save_file_type="csv")

    # Step 2: Preprocess data
    logging.info("Preprocessing data...")
    data_preprocessing = preprocessing.DataPreprocessor(config.SAMPLE_PATH)
    data_preprocessing.run(columns_to_drop=config.PREPROCESSING_PARAMS["columns_to_drop"],  missing_value_threshold=config.PREPROCESSING_PARAMS["missing_value_threshold"], imputation_strategy=config.PREPROCESSING_PARAMS["imputation_strategy"], knn_neighbors=config.PREPROCESSING_PARAMS["knn_neighbors"] )

    # # Step 3: Feature Engineering
    # logging.info("Generating features...")
    # df = feature_engineering.transform_features(df, config.FEATURE_PARAMS)

    # # Step 4: Train/Test Split
    # logging.info("Splitting dataset into train and test sets...")
    # X_train, X_test, y_train, y_test = preprocessing.split_data(df, config.TARGET_COLUMN, config.SPLIT_PARAMS)

    # # Step 5: Train Model
    # logging.info("Training model...")
    # model = train_model.train(X_train, y_train, config.MODEL_PARAMS)

    # # Step 6: Evaluate Model
    # logging.info("Evaluating model...")
    # results = evaluate_model.evaluate(model, X_test, y_test, config.METRIC_PARAMS)

    # # Step 7: Save results
    # logging.info("Saving model and results...")
    # train_model.save_model(model, config.MODEL_SAVE_PATH)
    # evaluate_model.save_results(results, config.RESULTS_PATH)

    # logging.info("Pipeline execution complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full data processing pipeline")
    parser.add_argument("--config", type=str, default="config.py", help="Path to the config file")
    args = parser.parse_args()

    main()

### Exemple of Pipeline
# import argparse
# import logging
# from src import data_loading, preprocessing, feature_engineering, train_model, evaluate_model, config

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# def main():
#     logging.info("Starting the data science pipeline...")

#     # Step 1: Load data
#     logging.info("Loading data...")
#     df = data_loading.load_data(config.DATA_PATH)

#     # Step 2: Preprocess data
#     logging.info("Preprocessing data...")
#     df = preprocessing.clean_data(df, config.PREPROCESSING_PARAMS)

#     # Step 3: Feature Engineering
#     logging.info("Generating features...")
#     df = feature_engineering.transform_features(df, config.FEATURE_PARAMS)

#     # Step 4: Train/Test Split
#     logging.info("Splitting dataset into train and test sets...")
#     X_train, X_test, y_train, y_test = preprocessing.split_data(df, config.TARGET_COLUMN, config.SPLIT_PARAMS)

#     # Step 5: Train Model
#     logging.info("Training model...")
#     model = train_model.train(X_train, y_train, config.MODEL_PARAMS)

#     # Step 6: Evaluate Model
#     logging.info("Evaluating model...")
#     results = evaluate_model.evaluate(model, X_test, y_test, config.METRIC_PARAMS)

#     # Step 7: Save results
#     logging.info("Saving model and results...")
#     train_model.save_model(model, config.MODEL_SAVE_PATH)
#     evaluate_model.save_results(results, config.RESULTS_PATH)

#     logging.info("Pipeline execution complete!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run the full data processing pipeline")
#     parser.add_argument("--config", type=str, default="config.py", help="Path to the config file")
#     args = parser.parse_args()

#     main()

