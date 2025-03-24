import pandas as pd
import os

path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(path, nrows=10000, sep='\t',encoding="utf-8",low_memory=True,na_filter=True)

df.to_csv("openfoodfacts_sample.csv", index=False, sep='\t', encoding="utf-8")

os.system("git add openfoodfacts_sample.csv")
os.system("git commit -m 'Ajout du fichier openfoodfacts_sample.csv'")
os.system("git push origin main")