import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.stats import zscore
import os

# Charger les données depuis un fichier CSV
file_path = "c:/Users/emili/OneDrive/Bureau/DP2/OpenFoodFact_GroupProject/data/dataset/openfoodfacts_sample.csv"
df = pd.read_csv(file_path, sep='\t', encoding="utf-8", low_memory=True)

# Sélectionner les colonnes numériques pour l'analyse des outliers
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Fonction 1 : Détection des outliers avec le critère de Tukey
def tukey_outliers(df, columns):
    outliers = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
    return outliers

# Fonction 2 : Détection des outliers avec le Z-score
def zscore_outliers(df, columns, threshold=3):
    outliers = {}
    for col in columns:
        df_cleaned = df[col].dropna()
        z_scores = zscore(df_cleaned)
        outliers[col] = df_cleaned.index[np.abs(z_scores) > threshold].tolist()
    return outliers

# Fonction 3 : Détection des outliers avec Isolation Forest
def isolation_forest_outliers(df, columns):
    outliers = {}
    model = IsolationForest(contamination=0.05, random_state=42)
    for col in columns:
        df_cleaned = df[[col]].dropna()
        if df_cleaned.empty:
            continue
        outliers[col] = df_cleaned[model.fit_predict(df_cleaned) == -1].index.tolist()
    return outliers

# Fonction 4 : Détection des outliers avec One-Class SVM
def one_class_svm_outliers(df, columns):
    outliers = {}
    model = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    for col in columns:
        df_cleaned = df[[col]].dropna()
        if df_cleaned.empty:
            continue
        outliers[col] = df_cleaned[model.fit_predict(df_cleaned) == -1].index.tolist()
    return outliers

# Fonction pour gérer les outliers en fonction de la stratégie choisie
def handle_outliers(df, outlier_indices, strategy="remove"):
    """
    Stratégie pour traiter les outliers :
    - "keep" : Conserver les outliers
    - "impute" : Remplacer les outliers par la médiane (pour les colonnes numériques uniquement)
    - "remove" : Supprimer les outliers
    """
    # Convertir les indices des outliers en liste
    outlier_indices = list(outlier_indices)

    if strategy == "remove":
        return df.drop(index=outlier_indices)
    
    elif strategy == "impute":
        # Limiter l'imputation aux colonnes numériques uniquement
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            median_value = df[col].median()
            df.loc[outlier_indices, col] = median_value
        return df
    
    elif strategy == "keep":
        return df
    
    else:
        print("Stratégie inconnue. La stratégie 'remove' sera utilisée par défaut.")
        return df.drop(index=outlier_indices)

# Détection des outliers avec les différentes méthodes
tukey_outliers_result = tukey_outliers(df, numeric_columns)
zscore_outliers_result = zscore_outliers(df, numeric_columns)
isolation_forest_outliers_result = isolation_forest_outliers(df, numeric_columns)
one_class_svm_outliers_result = one_class_svm_outliers(df, numeric_columns)

# Combiner tous les indices d'outliers détectés
all_outliers = set(sum(tukey_outliers_result.values(), [])) | \
               set(sum(zscore_outliers_result.values(), [])) | \
               set(sum(isolation_forest_outliers_result.values(), [])) | \
               set(sum(one_class_svm_outliers_result.values(), []))

# Appliquer chaque stratégie et sauvegarder les résultats
strategies = ["keep", "impute", "remove"]
for strategy in strategies:
    print(f"\n🔍 Application de la stratégie : {strategy}")
    df_processed = handle_outliers(df.copy(), all_outliers, strategy=strategy)
    
    # Afficher un aperçu des données après traitement
    print(f"Données après application de la stratégie '{strategy}':")
    print(df_processed.head())
    
    # Définir le chemin du fichier de sortie
    output_dir = "c:/Users/emili/OneDrive/Bureau/DP2/OpenFoodFact_GroupProject/data/dataset"
    output_file = f"{output_dir}/openfoodfacts_processed_{strategy}.csv"
    
    # Vérifier si le dossier existe
    if not os.path.exists(output_dir):
        print(f"❌ Le dossier spécifié n'existe pas : {output_dir}")
        exit(1)  # Arrêter le script si le dossier n'existe pas
    
    # Sauvegarder les résultats dans un fichier CSV
    df_processed.to_csv(output_file, index=False, sep='\t', encoding="utf-8")
    print(f"✅ Résultats sauvegardés dans '{output_file}'.")