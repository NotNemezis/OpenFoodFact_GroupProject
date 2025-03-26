import pandas as pd

def process_dataset_with_thresholds(input_file: str, output_dir: str):
    """
    Nettoie un dataset en supprimant les colonnes avec un pourcentage de NaN supérieur à un seuil
    et remplace les NaN restants par la médiane ou la moyenne. Génère des fichiers pour chaque seuil.

    Args:
        input_file: Chemin du fichier CSV nettoyé à charger.
        output_dir: Répertoire où sauvegarder les fichiers générés.
    """
    # Charger le dataset nettoyé
    print(f"Chargement des données depuis {input_file}...")
    df = pd.read_csv(input_file, sep='\t', encoding="utf-8")
    print(f"✅ Données chargées avec succès : {len(df)} lignes, {len(df.columns)} colonnes.")

    # Définir les seuils à tester
    thresholds = [60, 70, 80, 90]

    for threshold in thresholds:
        print(f"\n🔍 Traitement avec un seuil de valeurs manquantes de {threshold}%...")

        # Étape 1 : Supprimer les colonnes avec trop de NaN
        missing_percentages = df.isnull().mean() * 100  # Pourcentage de NaN par colonne
        columns_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
        print(f"🔍 Colonnes supprimées pour valeurs manquantes (> {threshold}%): {columns_to_drop}")
        df_threshold = df.drop(columns=columns_to_drop)

        # Étape 2 : Remplacer les NaN restants par la médiane
        df_median = df_threshold.fillna(df_threshold.median(numeric_only=True))
        output_file_median = f"{output_dir}/openfoodfacts_KNN_NAN_Median_{threshold}.csv"
        df_median.to_csv(output_file_median, index=False, sep='\t', encoding="utf-8")
        print(f"✅ Fichier généré avec médiane : {output_file_median}")

        # Étape 3 : Remplacer les NaN restants par la moyenne
        df_mean = df_threshold.fillna(df_threshold.mean(numeric_only=True))
        output_file_mean = f"{output_dir}/openfoodfacts_KNN_NAN_Mean_{threshold}.csv"
        df_mean.to_csv(output_file_mean, index=False, sep='\t', encoding="utf-8")
        print(f"✅ Fichier généré avec moyenne : {output_file_mean}")


if __name__ == "__main__":
    # Chemin du fichier nettoyé
    input_file = "c:/Users/emili/OneDrive/Bureau/DP2/OpenFoodFact_GroupProject/data/dataset/openfoodfacts_cleaned.csv"
    # Répertoire de sortie
    output_dir = "c:/Users/emili/OneDrive/Bureau/DP2/OpenFoodFact_GroupProject/data/dataset"

    # Lancer le traitement
    process_dataset_with_thresholds(input_file, output_dir)