from Tache0 import analyze_data_quality  # Importer uniquement la fonction nécessaire
import pandas as pd

def clean_dataset(df: pd.DataFrame, quality_report: dict) -> pd.DataFrame:
    """
    Nettoie un dataset en supprimant les colonnes inutiles et les lignes problématiques.

    Args:
        df: DataFrame pandas à nettoyer.
        quality_report: Rapport généré par la fonction `analyze_data_quality`.

    Returns:
        pd.DataFrame: DataFrame nettoyé.
    """
    # 1. Supprimer les colonnes spécifiques dès le début
    columns_to_remove = [
        "code", "url", "creator", "created_t", "created_datetime",
        "last_modified_t", "last_modified_datetime", "packaging", "packaging_tags",
        "brands_tags", "categories_tags", "categories_fr",
        "origins_tags", "manufacturing_places", "manufacturing_places_tags",
        "labels_tags", "labels_fr", "emb_codes", "emb_codes_tags",
        "first_packaging_code_geo", "cities", "cities_tags", "purchase_places",
        "countries_tags", "countries_fr", "image_ingredients_url",
        "image_ingredients_small_url", "image_nutrition_url", "image_nutrition_small_url",
        "image_small_url", "image_url", "last_updated_t", "last_updated_datetime", "last_modified_by"
    ]
    columns_to_remove = [col for col in columns_to_remove if col in df.columns]  # Vérifier l'existence
    print(f"🔍 Colonnes supprimées spécifiquement : {columns_to_remove}")
    df = df.drop(columns=columns_to_remove)

    # Supprimer les colonnes contenant 100% de valeurs manquantes
    df = df.dropna(axis=1, how="all")
    print("🔍 Colonnes contenant 100% de valeurs manquantes supprimées.")

    # Supprimer les lignes contenant 100% de valeurs manquantes
    df = df.dropna(axis=0, how="all")
    print("🔍 Lignes contenant 100% de valeurs manquantes supprimées.")

    # 2. Supprimer les colonnes constantes
    constant_columns = quality_report['quality_issues']['constant_columns']
    constant_columns = [col for col in constant_columns if col in df.columns]  # Vérifier l'existence
    print(f"🔍 Colonnes constantes supprimées : {constant_columns}")
    df = df.drop(columns=constant_columns)

    # 3. Supprimer les colonnes numériques à faible variance
    low_variance_columns = quality_report['quality_issues']['low_variance_columns']
    low_variance_columns = [col for col in low_variance_columns if col in df.columns]  # Vérifier l'existence
    print(f"🔍 Colonnes numériques à faible variance supprimées : {low_variance_columns}")
    df = df.drop(columns=low_variance_columns)

    # 4. Supprimer les colonnes avec une forte cardinalité
    high_cardinality_columns = quality_report['quality_issues']['high_cardinality_columns']
    high_cardinality_columns = [col for col in high_cardinality_columns if col in df.columns]  # Vérifier l'existence
    print(f"🔍 Colonnes avec forte cardinalité supprimées : {high_cardinality_columns}")
    df = df.drop(columns=high_cardinality_columns)

    # 5. Supprimer les doublons
    initial_row_count = len(df)
    df = df.drop_duplicates()
    print(f"🔍 Lignes doublons supprimées : {initial_row_count - len(df)}")

    return df

if __name__ == "__main__":
    # Charger les données directement depuis un fichier CSV
    input_file = "c:/Users/emili/OneDrive/Bureau/DP2/OpenFoodFact_GroupProject/data/dataset/openfoodfacts_sample.csv"
    print(f"Chargement des données depuis {input_file}...")
    df = pd.read_csv(input_file, sep='\t', encoding="utf-8")
    print(f"✅ Données chargées avec succès : {len(df)} lignes, {len(df.columns)} colonnes.")

    # Analyse de la qualité des données
    print("\n🔍 Analyse de la qualité des données")
    quality_report = analyze_data_quality(df)  # Utiliser la fonction de Tache0

    # Nettoyer le dataset
    print("\n🔍 Nettoyage du dataset...")
    cleaned_df = clean_dataset(df, quality_report)

    # Sauvegarder le dataset nettoyé
    output_file = "c:/Users/emili/OneDrive/Bureau/DP2/OpenFoodFact_GroupProject/data/dataset/openfoodfacts_cleaned.csv"
    cleaned_df.to_csv(output_file, index=False, sep='\t', encoding="utf-8")
    print(f"✅ Dataset nettoyé sauvegardé dans : {output_file}")