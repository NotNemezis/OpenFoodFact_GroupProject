import os
import pandas as pd
from Tache3 import handle_outliers, tukey_outliers, zscore_outliers, isolation_forest_outliers, one_class_svm_outliers

# Répertoires
base_dir = "c:/Users/emili/OneDrive/Bureau/DP2/OpenFoodFact_GroupProject/data/dataset"  # Répertoire des fichiers d'entrée
output_dir = "c:/Users/emili/OneDrive/Bureau/DP2/OpenFoodFact_GroupProject/data/processed"  # Répertoire de sortie existant

# Vérifier si le répertoire de sortie existe
if not os.path.exists(output_dir):
    print(f"❌ Le répertoire de sortie n'existe pas : {output_dir}")
    exit(1)  # Arrêter le script si le répertoire n'existe pas

# Liste des fichiers générés par la tâche 2
files_to_process = [
    "openfoodfacts_KNN_NAN_Mean_60.csv",
    "openfoodfacts_KNN_NAN_Mean_70.csv",
    "openfoodfacts_KNN_NAN_Mean_80.csv",
    "openfoodfacts_KNN_NAN_Mean_90.csv",
    "openfoodfacts_KNN_NAN_Median_60.csv",
    "openfoodfacts_KNN_NAN_Median_70.csv",
    "openfoodfacts_KNN_NAN_Median_80.csv",
    "openfoodfacts_KNN_NAN_Median_90.csv"
]

# Stratégies pour les outliers
strategies = ["keep", "impute", "remove"]

# Parcourir les fichiers et appliquer les stratégies
for file_name in files_to_process:
    file_path = os.path.join(base_dir, file_name)
    
    # Vérifier si le fichier existe
    if not os.path.exists(file_path):
        print(f"❌ Fichier introuvable : {file_name}")
        continue

    print(f"🔍 Traitement du fichier : {file_name}")
    
    # Charger le fichier CSV
    df = pd.read_csv(file_path, sep='\t', encoding="utf-8", low_memory=True)

    # Sélectionner les colonnes numériques pour l'analyse des outliers
    numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()

    # Détection des outliers avec les méthodes définies dans Tache3
    tukey_outliers_result = tukey_outliers(df, numeric_columns)
    zscore_outliers_result = zscore_outliers(df, numeric_columns)
    isolation_forest_outliers_result = isolation_forest_outliers(df, numeric_columns)
    one_class_svm_outliers_result = one_class_svm_outliers(df, numeric_columns)

    # Combiner tous les indices d'outliers détectés
    all_outliers = set(sum(tukey_outliers_result.values(), [])) | \
                   set(sum(zscore_outliers_result.values(), [])) | \
                   set(sum(isolation_forest_outliers_result.values(), [])) | \
                   set(sum(one_class_svm_outliers_result.values(), []))

    # Appliquer chaque stratégie
    for strategy in strategies:
        print(f"  ➡️ Application de la stratégie : {strategy}")
        df_processed = handle_outliers(df.copy(), all_outliers, strategy=strategy)

        # Générer le nom du fichier de sortie
        output_file_name = f"{os.path.splitext(file_name)[0]}_{strategy}.csv"
        output_file_path = os.path.join(output_dir, output_file_name)

        # Sauvegarder le fichier
        df_processed.to_csv(output_file_path, index=False, sep='\t', encoding="utf-8")
        print(f"  ✅ Fichier généré : {output_file_path}")