# 📊 Projet Open Food Facts - Nettoyage et Clustering des Données  

## 🏆 Objectif  
Ce projet vise à exploiter la base de données **Open Food Facts** afin de :  
✅ Nettoyer et préparer les données efficacement 📌  
✅ Appliquer des techniques de **scaling** et d’**encodage** adaptées 🎛️  
✅ Réaliser du **clustering** pour identifier des groupes de produits similaires 🔍  

## 📂 Structure du projet  

OPENFOODFACT_GROUPPROJECT/
│── data/
│   ├── dataset
│   │   ├── en.openfoodfacts.org.products.csv
│   │   └── sample_10000.csv
│   ├── processed
│   │   └── preprocessed_sample_10000.csv
│   └── results
│── notebooks/
│── src/
│   ├── __init__.py # Permet de traiter `src` comme un package Python
│   ├── data_loading.py 
│   ├── preprocessing.py          
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── experiment_logger.py
│   ├── config.py
│   └── run_pipeline.py
│── scripts/
│   └── __init__.py             
│── requirements.txt
│── README.md

## Get started

python src/run_pipeline.py