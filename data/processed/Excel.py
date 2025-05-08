import pandas as pd

# Données du tableau
data = {
    "Fichier": [
        "openfoodfactstrain_KNN(mean)_60_impute.csv",
        "openfoodfactstrain_KNN(mean)_60_remove.csv",
        "openfoodfactstrain_KNN(mean)_70_impute.csv",
        "openfoodfactstrain_KNN(mean)_70_remove.csv",
        "openfoodfactstrain_KNN(mean)_80_impute.csv",
        "openfoodfactstrain_KNN(mean)_80_remove.csv",
        "openfoodfactstrain_KNN(mean)_90_impute.csv",
        "openfoodfactstrain_KNN(mean)_90_remove.csv",
        "openfoodfactstrain_NAN_Median_60_impute.csv",
        "openfoodfactstrain_NAN_Median_60_remove.csv",
        "openfoodfactstrain_NAN_Median_70_impute.csv",
        "openfoodfactstrain_NAN_Median_70_remove.csv",
        "openfoodfactstrain_NAN_Median_80_impute.csv",
        "openfoodfactstrain_NAN_Median_80_remove.csv",
        "openfoodfactstrain_NAN_Median_90_impute.csv",
        "openfoodfactstrain_NAN_Median_90_remove.csv",
        "openfoodfactstrain_NAN_Mean_60_impute.csv",
        "openfoodfactstrain_NAN_Mean_60_remove.csv",
        "openfoodfactstrain_NAN_Mean_70_impute.csv",
        "openfoodfactstrain_NAN_Mean_70_remove.csv",
        "openfoodfactstrain_NAN_Mean_80_impute.csv",
        "openfoodfactstrain_NAN_Mean_80_remove.csv",
        "openfoodfactstrain_NAN_Mean_90_impute.csv",
        "openfoodfactstrain_NAN_Mean_90_remove.csv",
    ],
    "KNN (Mean)": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Median": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "Mean": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    "Threshold 60": [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    "Threshold 70": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    "Threshold 80": [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    "Threshold 90": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    "Impute": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "Remove": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
}

# Créer un DataFrame
df = pd.DataFrame(data)

# Sauvegarder en fichier Excel
output_file = "c:/Users/emili/OneDrive/Bureau/DP2/OpenFoodFact_GroupProject/data/processed/methods_summary.xlsx"
df.to_excel(output_file, index=False)

print(f"✅ Tableau sauvegardé en Excel : {output_file}")