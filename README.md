# 📊 Projet Open Food Facts - Nettoyage et Clustering des Données  

## 🏆 Objectif  
Ce projet exploite la base de données **Open Food Facts** afin de :  
- ✅ **Nettoyer** et **préparer** les données efficacement 📌  
- ✅ Appliquer des techniques de **scaling** et d’**encodage** adaptées 🎛️  
- ✅ Réaliser du **clustering** pour identifier des groupes de produits similaires 🔍  

---

## 📂 Structure du projet  

```
OPENFOODFACT_GROUPPROJECT/
│── data/
│   ├── dataset/
│   │   ├── en.openfoodfacts.org.products.csv
│   │   └── sample_10000.csv
│   ├── processed/
│   │   └── preprocessed_sample_10000.csv
│   └── results/
│── notebooks/
│── src/
│   ├── __init__.py             # Permet de traiter `src` comme un package Python
│   ├── data_loading.py         # Chargement des données
│   ├── preprocessing.py        # Nettoyage et prétraitement des données
│   ├── feature_engineering.py  # Création de nouvelles variables
│   ├── train_model.py          # Entraînement des modèles de clustering
│   ├── evaluate_model.py       # Évaluation des modèles
│   ├── experiment_logger.py    # Gestion des logs et des expériences
│   ├── config.py               # Fichier de configuration du projet
│   └── run_pipeline.py         # Script principal d'exécution du pipeline
│── scripts/
│   └── __init__.py             
│── requirements.txt            # Liste des dépendances
│── README.md                   # Documentation du projet
```

---

## 🚀 Installation & Utilisation  

### 📥 Prérequis  
Assurez-vous d'avoir **Python 3.x** installé ainsi que `pip`. Vous pouvez
