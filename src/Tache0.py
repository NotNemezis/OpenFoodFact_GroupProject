import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.stats import zscore

# Charger les données depuis le fichier openfoodfacts_sample.csv
def load_data(file_path: str = "c:/Users/emili/OneDrive/Bureau/DP2/OpenFoodFact_GroupProject/data/dataset/openfoodfacts_sample.csv") -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV.

    Args:
        file_path: Chemin du fichier CSV à charger.
 
    Returns:
        pd.DataFrame: Le DataFrame chargé.
    """
    print(f"Chargement des données depuis {file_path}...")
    df = pd.read_csv(file_path, sep='\t', encoding="utf-8", low_memory=True)
    print(f"✅ Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return df
#Détection des types de variables 
def detect_variable_types(df: pd.DataFrame) -> Dict:
    """
    Détecte les types de variables dans un DataFrame : numériques, catégorielles nominales, ordinales.

    Args:
        df: DataFrame pandas à analyser.

    Returns:
        Dict: Dictionnaire contenant les colonnes par type.
    """
    # Colonnes numériques
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Colonnes catégorielles
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Catégories prédéfinies pour les colonnes ordinales
    predefined_ordinal_categories = {
        "nutri_score": ["e", "d", "c", "b", "a"]
    }

    ordinal_columns = []
    non_ordinal_columns = []

    # Identifier les colonnes ordinales et non ordinales
    for col in categorical_columns:
        unique_values = df[col].dropna().astype(str).str.lower().str.strip().unique()
        
        for predefined_values in predefined_ordinal_categories.values():
            matching_values = [val for val in unique_values if val in predefined_values]
            if len(matching_values) / len(unique_values) > 0.7:  # Si 70% des valeurs correspondent
                ordinal_columns.append(col)
                break
        else:
            non_ordinal_columns.append(col)

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "ordinal_columns": ordinal_columns,
        "non_ordinal_columns": non_ordinal_columns
    }

# Analyse de la qualité des données
def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    Analyse la qualité des données dans un DataFrame : dimensions, types de données,
    valeurs manquantes, valeurs uniques, colonnes constantes et utilisation mémoire.

    Args:
        df: DataFrame pandas à analyser
        
    Returns:
        Dict: Rapport contenant les résultats de l'analyse
    """
    n_rows, n_cols = df.shape  # Dimensions du DataFrame
    dtypes = df.dtypes.value_counts()  # Comptage des types de données
    
    # Comptage des valeurs uniques par colonne
    unique_counts = df.nunique()
    unique_percentages = (unique_counts / n_rows * 100).round(2)
    
    # Colonnes constantes et avec trop de catégories
    constant_columns = [col for col in df.columns if unique_counts[col] == 1]
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    high_cardinality_columns = [col for col in categorical_columns 
                                if unique_counts[col] / n_rows > 0.5]
    
    # Colonnes numériques à faible variance
    numeric_columns = df.select_dtypes(include=np.number).columns
    low_variance_columns = [col for col in numeric_columns 
                            if unique_counts[col] / n_rows < 0.01]
    
    # Utilisation de la mémoire
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum() / 1024**2  # Mémoire totale en Mo
    
    return {
        'dimensions': {'rows': n_rows, 'columns': n_cols},
        'dtypes': dtypes.to_dict(),
        'unique_values': {
            'counts': unique_counts.to_dict(),
            'percentages': unique_percentages.to_dict()
        },
        'quality_issues': {
            'constant_columns': constant_columns,
            'high_cardinality_columns': high_cardinality_columns,
            'low_variance_columns': low_variance_columns
        },
        'memory_usage': {
            'total_mb': total_memory,
            'per_column': memory_usage.to_dict()
        }
    }

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les données
    df = load_data()

    # Détection des types de variables
    print("\n🔍 Détection des types de variables")
    variable_types = detect_variable_types(df)

    # Afficher les résultats de la détection
    print(f"Colonnes numériques : {variable_types['numeric_columns']}")
    print(f"Colonnes catégorielles : {variable_types['categorical_columns']}")
    print(f"Colonnes ordinales : {variable_types['ordinal_columns']}")
    print(f"Colonnes non ordinales : {variable_types['non_ordinal_columns']}")

    # Analyse de la qualité des données
    print("\n🔍 Analyse de la qualité des données")
    quality_report = analyze_data_quality(df)

    # Affichage structuré des résultats
    print("\n📊 Résultats de l'analyse de la qualité des données :")

    # Dimensions
    print("\n🔹 Dimensions :")
    print(f"Nombre de lignes : {quality_report['dimensions']['rows']}")
    print(f"Nombre de colonnes : {quality_report['dimensions']['columns']}")

    # Types de données
    print("\n🔹 Types de données :")
    for dtype, count in quality_report['dtypes'].items():
        print(f"{dtype}: {count}")

    

    # Valeurs uniques
    print("\n🔹 Valeurs uniques :")
    print("Nombre de valeurs uniques par colonne :")
    for col, count in quality_report['unique_values']['counts'].items():
        percentage = quality_report['unique_values']['percentages'][col]
        print(f"- {col} : {count} valeurs uniques ({percentage}%)")

    # Problèmes de qualité
    print("\n🔹 Problèmes de qualité :")
    print(f"Colonnes constantes : {quality_report['quality_issues']['constant_columns']}")
    print(f"Colonnes avec une forte cardinalité : {quality_report['quality_issues']['high_cardinality_columns']}")
    print(f"Colonnes numériques à faible variance : {quality_report['quality_issues']['low_variance_columns']}")

    # Utilisation de la mémoire
    print("\n🔹 Utilisation de la mémoire :")
    print(f"Utilisation totale de la mémoire : {quality_report['memory_usage']['total_mb']:.2f} Mo")
    print("Utilisation de la mémoire par colonne :")
    for col, memory in quality_report['memory_usage']['per_column'].items():
        print(f"- {col} : {memory / 1024:.2f} Ko")

    
