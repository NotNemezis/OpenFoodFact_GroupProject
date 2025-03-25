import pandas as pd

# URL du dataset
path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"

# Charger le dataset en entier pour tirer un échantillon aléatoire, en ignorant les lignes problématiques
df = pd.read_csv(
    path,
    sep='\t',
    encoding="utf-8",
    low_memory=True,
    na_filter=True,
    on_bad_lines='skip'  # Ignorer les lignes problématiques
)

# Tirer 10 000 lignes aléatoires
df_sample = df.sample(n=10000, random_state=42)

# Sauvegarder l'échantillon
df_sample.to_csv("openfoodfacts_sample.csv", index=False, sep='\t', encoding="utf-8")

print("✅ Échantillon aléatoire de 10 000 lignes sauvegardé avec succès !")
