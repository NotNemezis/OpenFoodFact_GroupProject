import pandas as pd
from scripts.script_pipeline_feature_selection import feature_selection_pipeline

df = pd.read_csv("", sep=",")

new_df = feature_selection_pipeline(df, target_column="target", method="correlation", threshold=0.9)

pd.to_csv(new_df, sep=",", index=False)