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

## 👥 Répartition des tâches  

Le projet a été réalisé en collaboration avec une équipe de 4 personnes, chacun ayant contribué à des étapes spécifiques de la chaîne de traitement :  

### 🔄 **Pipeline global**  
- **Eric** : Responsable du développement global de la pipeline.  
    - 🛠️ A également pris en charge la majeure partie du **prétraitement des données** :  
        - Séparation des données selon leurs types (numériques et non-numériques).  
        - Encodage des données non-numériques en fonction de leur cardinalité.  

### 📉 **Sous-échantillonnage des données**  
- **Emilia** : Chargée de la génération de différents sous-échantillons du dataset initial :  
    - 📊 Définition de seuils de tolérance (60% à 90%) pour les valeurs contenues dans les échantillons.  
    - 🧹 Application de 3 méthodes de remplacement des valeurs manquantes :  
        - Imputation par **KNN**.  
        - Remplacement par la **moyenne**.  
        - Remplacement par la **médiane**.  
    - 🚦 Gestion des outliers selon 3 stratégies :  
        - Suppression des outliers.  
        - Conservation des outliers.  
        - Remplacement des outliers en utilisant la même méthode d’imputation que celle appliquée au dataset.  

### 🧬 **Sélection des variables**  
- **Grégoire** : Responsable de la **sélection des features** pertinentes pour le modèle.  

### 🤖 **Entraînement et évaluation des modèles**  
- **Alexandre** : Chargé de l’**entraînement des modèles** de clustering et de leur **évaluation**.  

---  
Grâce à cette répartition, chaque membre a pu se concentrer sur une étape clé, garantissant une progression fluide et collaborative du projet.  
**
**