# import pandas as pd
# import os
# import json
# from datetime import datetime

# EXPERIMENT_LOG_PATH = "data/results/experiments_log.csv"

# def log_experiment(config, results):
#     """
#     Logs the parameters and results of a pipeline run into a CSV file.

#     Parameters:
#     - config (dict): Dictionary of parameters used in the pipeline.
#     - results (dict): Dictionary of model performance metrics.

#     Returns:
#     - None (logs the data in a CSV file)
#     """
#     log_data = {
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "data_path": config["DATA_PATH"],
#         "model_type": config["MODEL_PARAMS"]["model_type"],
#         "hyperparameters": json.dumps(config["MODEL_PARAMS"]["hyperparameters"]),
#         "train_size": config["SPLIT_PARAMS"]["train_size"],
#         "stratified": config["SPLIT_PARAMS"]["stratified"],
#         "metrics": json.dumps(results)
#     }

#     # Check if log file exists
#     if not os.path.exists(EXPERIMENT_LOG_PATH):
#         df = pd.DataFrame([log_data])
#     else:
#         df = pd.read_csv(EXPERIMENT_LOG_PATH)
#         df = df.append(log_data, ignore_index=True)

#     df.to_csv(EXPERIMENT_LOG_PATH, index=False)
#     print(f"Experiment logged in {EXPERIMENT_LOG_PATH}")
