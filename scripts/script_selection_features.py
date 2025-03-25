class selection_feature:
    """
    Pipeline pour la sélection des features.
    Arguments :
        df (pd.DataFrame) : DataFrame à partir duquel les features seront sélectionnées.
    """
    def select_highly_correlated_features(self, threshold=0.9):
        """
        Sélectionne les features les plus corrélées entre elles et les élimine si leur corrélation dépasse un certain seuil.

        Arguments :
            threshold (float) : float - Seuil de corrélation au-delà duquel les features seront éliminées.

        Retour :
            pd.DataFrame : Un DataFrame avec les features sélectionnées.
        """
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