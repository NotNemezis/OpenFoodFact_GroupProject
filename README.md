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

# 📊 Projet Open Food Facts - Nettoyage et Clustering des Données  

## 🚀 Comment lancer le projet ?

Pour lancer le projet, suivez les étapes ci-dessous :

1. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   ```

2. Configurez les paramètres dans le fichier src/config.py. Ce fichier contient toutes les options et paramètres nécessaires pour chaque étape de la pipeline.

3. Exécutez la pipeline avec la commande suivante :
```bash
python src/run_pipeline.py
```

🛠️ Pipeline de traitement des données
La pipeline est décomposée en 7 étapes principales. Actuellement, seules les 3 premières étapes sont configurées :

Étape 1 : Chargement des données
Chargement des données brutes depuis le chemin spécifié dans config.DATA_PATH.
Sous-échantillonnage des données (par exemple, 10 000 lignes) et sauvegarde dans un fichier CSV.

Étape 2 : Prétraitement des données
Suppression des colonnes inutiles.
Gestion des valeurs manquantes (imputation par moyenne, médiane, ou KNN).
Application des paramètres définis dans config.PREPROCESSING_PARAMS.

Étape 3 : Ingénierie des features
Sélection des features pertinentes en fonction de la méthode spécifiée dans config.FEATURE_PARAMS.
Méthodes disponibles : correlation, variance, anova_k_best, random_forest, etc.

📋 Exemple de configuration dans src/config.py
Voici un exemple de configuration pour les paramètres de la pipeline :

```python
DATA_PATH = "./data/dataset/openfoodfacts.csv"
SAMPLE_PATH = "./data/dataset/sample_10000.csv"
DATA_PROCESSED_PATH = "./data/processed/processed_sample_10000.csv"

PREPROCESSING_PARAMS = {
    "columns_to_drop": ["column1", "column2"],
    "missing_value_threshold": 0.3,
    "imputation_strategy": "mean",
    "knn_neighbors": 5,
}

FEATURE_PARAMS = {
    "method": "correlation",
    "threshold": 0.9,
    "target_column": "target",
    "k": 10,
    "percentile": 10,
    "score_func": None,
    "mode": "percentile",
    "param": 10,
    "threshold_model": "mean",
    "n_features_to_select": 10,
    "direction": "forward",
}
```

📈 Étapes futures
Les étapes suivantes restent à configurer dans la pipeline :

Étape 4 : Train/Test Split : Division des données en ensembles d'entraînement et de test.
Étape 5 : Entraînement du modèle : Entraînement des modèles de clustering ou de classification.
Étape 6 : Évaluation du modèle : Évaluation des performances des modèles.
Étape 7 : Sauvegarde des résultats : Sauvegarde des modèles et des résultats d'évaluation.