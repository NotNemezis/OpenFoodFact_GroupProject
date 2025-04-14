import logging
from script_selection_features import (
    CorrelationFeatureSelector,
    VarianceFeatureSelector,
    AnovaFeatureSelector,
    GenericUnivariateFeatureSelector,
    ModelFeatureSelector,
    SequentialFeatureSelector
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def feature_selection_pipeline(df, target_column, method, **kwargs):
    logging.info(f"Starting feature selection using method: {method}")

    selector = None

    if method == "correlation":
        selector = CorrelationFeatureSelector(df)
        return selector.select_highly_correlated_features(**kwargs)
    
    elif method == "variance":
        selector = VarianceFeatureSelector(df)
        return selector.select_features_by_variance(**kwargs)
    
    elif method in ["anova_k_best", "anova_percentile"]:
        selector = AnovaFeatureSelector(df)
        if method == "anova_k_best":
            return selector.select_k_best_features(target_column=target_column, **kwargs)
        else:
            return selector.select_percentile_best_features(target_column=target_column, **kwargs)
    
    elif method == "generic_univariate":
        selector = GenericUnivariateFeatureSelector(df)
        return selector.select_generic_univariate_features(target_column=target_column, **kwargs)
    
    elif method in ["model_based", "lasso"]:
        selector = ModelFeatureSelector(df)
        if method == "model_based":
            return selector.select_from_model_features(target_column=target_column, **kwargs)
        else:
            return selector.select_lasso_features(target_column=target_column, **kwargs)
    
    elif method == "sequential":
        selector = SequentialFeatureSelector(df)
        return selector.select_sequential_features(target_column=target_column, **kwargs)
    
    else:
        logging.error(f"Unknown method: {method}")
        raise ValueError(f"Unknown method: {method}")