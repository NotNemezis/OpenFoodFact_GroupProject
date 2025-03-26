from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectPercentile, GenericUnivariateSelect, SelectFromModel, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from statsmodels.tools import add_constant
import statsmodels.api as sm

class CorrelationFeatureSelector:
    def __init__(self, df):
        self.df = df

    def select_highly_correlated_features(self, threshold=0.9):
        corr_matrix = self.df.corr(numeric_only=True)
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

class VarianceFeatureSelector:
    def __init__(self, df):
        self.df = df

    def select_features_by_variance(self, threshold=0.8):
        X = self.df.select_dtypes(include=["int64", "float64"])
        selector = VarianceThreshold(threshold=(threshold * (1 - threshold)))
        selector.fit_transform(X)
        selected_columns = X.columns[selector.get_support()]
        X_selected_df = X[selected_columns]
        print(f"Features sélectionnées : {selected_columns.tolist()}")
        print(f"Nouvelle forme du DataFrame : {X_selected_df.shape}")
        return X_selected_df

class AnovaFeatureSelector:
    def __init__(self, df):
        self.df = df

    def select_k_best_features(self, target_column, k=10):
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
        X = self.df.select_dtypes(include=["int64", "float64"])
        y = self.df[target_column]
        selector = SelectPercentile(score_func=f_classif, percentile=percentile)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        X_selected_df = X[selected_columns]
        print(f"Features sélectionnées : {selected_columns.tolist()}")
        print(f"Nouvelle forme du DataFrame : {X_selected_df.shape}")
        return X_selected_df

class GenericUnivariateFeatureSelector:
    def __init__(self, df):
        self.df = df

    def select_generic_univariate_features(self, target_column, mode='percentile', param=10):
        X = self.df.select_dtypes(include=["int64", "float64"])
        y = self.df[target_column]
        selector = GenericUnivariateSelect(score_func=f_classif, mode=mode, param=param)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        X_selected_df = X[selected_columns]
        print(f"Features sélectionnées : {selected_columns.tolist()}")
        print(f"Nouvelle forme du DataFrame : {X_selected_df.shape}")
        return X_selected_df

class ModelFeatureSelector:
    def __init__(self, df):
        self.df = df

    def select_from_model_features(self, target_column, threshold='mean', max_features=None):
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

class SequentialFeatureSelector:
    def __init__(self, df):
        self.df = df

    def select_sequential_features(self, target_column, n_features_to_select=10, direction='forward'):
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