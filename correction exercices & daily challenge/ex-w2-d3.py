#!/usr/bin/env python3
"""
Author : Clara Martinez

Script complet pour Data Preprocessing et Transformation,
contenant des exercices de nettoyage, transformation, fusion, réduction dimensionnelle, agrégation, et visualisation.

Avant exécution, assure-toi d’avoir installé les packages nécessaires, par exemple :
    pip install pandas numpy matplotlib seaborn scikit-learn requests
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------------------------------------------------------
# GREEN EXERCISES : KNOWLEDGE CHECKS & OUTLIER IDENTIFICATION
# -----------------------------------------------------------------------------
def green_exercises():
    print("\n=== GREEN EXERCISES ===")
    # Knowledge Checks (affichage de réponses)
    print("\nKnowledge Check - Remplir des valeurs manquantes par 0 :")
    print("→ Quand les valeurs manquantes ne sont pas informatives et peuvent être remplacées sans perte d'information.")
    
    print("\nKnowledge Check - Retirer des lignes avec données manquantes :")
    print("→ Quand les valeurs manquantes se concentrent dans certaines colonnes et que ces lignes sont minoritaires.")
    
    # Challenge : Identify Outliers sur la colonne 'Age' dans Titanic.
    try:
        titanic = pd.read_csv("titanic.csv")
        age = titanic['Age'].dropna()  # on ignore les NaN
        Q1 = age.quantile(0.25)
        Q3 = age.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        # Si lower_bound est négatif, on le règle à 0
        if lower_bound < 0:
            lower_bound = 0
        upper_bound = Q3 + 1.5 * IQR
        outliers = age[(age < lower_bound) | (age > upper_bound)]
        print("\nChallenge Outliers sur 'Age' (Titanic) :")
        print(f"Q1  = {Q1}\nQ3  = {Q3}\nIQR = {IQR}\nLower Bound = {lower_bound}\nUpper Bound = {upper_bound}")
        print(f"Nombre d'outliers : {outliers.shape[0]}")
        if outliers.shape[0] == 0:
            print("→ Aucune valeur anormale détectée (tous les âges sont dans l'intervalle [0, {upper_bound}]).")
    except Exception as e:
        print("Erreur lors du calcul des outliers sur Titanic:", e)
    
    print("\nKnowledge Check - Min-Max normalization :")
    print("→ Elle permet de ramener les valeurs entre 0 et 1 afin de mettre sur un pied d'égalité des données sur des échelles différentes.")
    
    print("\nKnowledge Check - PCA à 2 composantes :")
    print("→ 'reduced_data' correspond aux données d'origine projetées dans un espace à 2 dimensions, conservant l'essentiel de l'information.")
    
    print("\nKnowledge Check - Agrégation de données :")
    print("→ Utile pour résumer de gros volumes de données, par exemple pour obtenir des moyennes par groupe ou pour lisser des séries temporelles.")

    print("\nKnowledge Check - Clé commune pour merger Titanic et données démographiques :")
    print("→ 'PassengerId'\n")
    
# -----------------------------------------------------------------------------
# CHALLENGE : Short-term Daily Precipitation Forecasting
# -----------------------------------------------------------------------------
def precipitation_forecasting():
    print("\n=== SHORT-TERM DAILY PRECIPITATION FORECASTING ===")
    # Téléchargement via Kaggle (ici, on commente ces commandes car elles sont à exécuter dans Colab)
    # Par exemple :
    # from google.colab import files
    # files.upload()
    # !mkdir -p ~/.kaggle
    # !cp kaggle.json ~/.kaggle/
    # !chmod 600 ~/.kaggle/kaggle.json
    # !kaggle datasets download -d muthuj7/weather-dataset --unzip
    
    # On charge le dataset (assure-toi que le fichier 'weather_dataset.csv' existe)
    try:
        weather_df = pd.read_csv("weather_dataset.csv")
        print("\nAperçu du dataset Weather :")
        print(weather_df.head())
        print("\nInfo du dataset :")
        print(weather_df.info())
        print("\nStatistiques descriptives :")
        print(weather_df.describe())
    except FileNotFoundError:
        print("Fichier 'weather_dataset.csv' non trouvé. Télécharge-le depuis Kaggle et place-le ici.")
    
    # Normalisation de la variable 'Precipitation'
    try:
        scaler = MinMaxScaler()
        weather_df['Precipitation_normalized'] = scaler.fit_transform(weather_df[['Precipitation']])
    except Exception as e:
        print("Erreur lors de la normalisation :", e)
    
    # Réduction de dimension avec PCA sur toutes les colonnes (sauf Precipitation_normalized)
    try:
        pca = PCA(n_components=2)
        # On sélectionne seulement les colonnes numériques, en excluant 'Precipitation_normalized'
        numerical_cols = weather_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if 'Precipitation_normalized' in numerical_cols:
            numerical_cols.remove('Precipitation_normalized')
        reduced_data = pca.fit_transform(weather_df[numerical_cols])
        print("\nAprès PCA, 'reduced_data' a la forme :", reduced_data.shape)
    except Exception as e:
        print("Erreur lors du PCA :", e)
    
    # Agrégation des précipitations par Location (moyenne)
    try:
        aggregated_data = weather_df.groupby('Location')['Precipitation'].mean().reset_index()
        print("\nAperçu de l'agrégation (moyenne de précipitations par Location) :")
        print(aggregated_data.head())
    except Exception as e:
        print("Erreur lors de l'agrégation :", e) 

# -----------------------------------------------------------------------------
# EXERCISES XP
# -----------------------------------------------------------------------------
def exercise_duplicate_removal():
    print("\n=== EXERCISE XP 1 : Duplicate Detection And Removal ===")
    try:
        titanic_df = pd.read_csv("titanic.csv")
        print(f"Nombre initial de lignes : {titanic_df.shape[0]}")
        duplicates = titanic_df.duplicated()
        print(f"Nombre de doublons détectés : {duplicates.sum()}")
        titanic_df = titanic_df.drop_duplicates()
        print(f"Nombre de lignes après suppression des doublons : {titanic_df.shape[0]}")
    except FileNotFoundError:
        print("Fichier 'titanic.csv' introuvable.")

def exercise_handling_missing():
    print("\n=== EXERCISE XP 2 : Handling Missing Values ===")
    try:
        titanic_df = pd.read_csv("titanic.csv")
        missing_values = titanic_df.isnull().sum()
        print("Valeurs manquantes par colonne :\n", missing_values)
        # Imputation de 'Age' par la médiane
        imputer = SimpleImputer(strategy='median')
        titanic_df['Age'] = imputer.fit_transform(titanic_df[['Age']])
        # Pour 'Embarked' : utiliser la modalité la plus fréquente
        most_common = titanic_df['Embarked'].mode()[0]
        titanic_df['Embarked'].fillna(most_common, inplace=True)
        # Pour 'Cabin', remplacer par 'Unknown'
        titanic_df['Cabin'].fillna("Unknown", inplace=True)
        print("Après imputation :\n", titanic_df.isnull().sum())
    except FileNotFoundError:
        print("Fichier 'titanic.csv' introuvable.")

def exercise_feature_engineering():
    print("\n=== EXERCISE XP 3 : Feature Engineering ===")
    try:
        titanic_df = pd.read_csv("titanic.csv")
        # Création de la variable Family_Size
        titanic_df['Family_Size'] = titanic_df['SibSp'] + titanic_df['Parch']
        # Extraction du Title depuis le nom
        titanic_df['Title'] = titanic_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        # Transformation des variables catégorielles en variables binaires
        titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked', 'Title'])
        print("Aperçu après feature engineering :")
        print(titanic_df.head())
    except FileNotFoundError:
        print("Fichier 'titanic.csv' introuvable.")

def exercise_outlier_detection():
    print("\n=== EXERCISE XP 4 : Outlier Detection And Handling ===")
    try:
        titanic_df = pd.read_csv("titanic.csv")
        # Détection sur 'Fare'
        Q1 = titanic_df['Fare'].quantile(0.25)
        Q3 = titanic_df['Fare'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        initial_count = titanic_df.shape[0]
        titanic_df = titanic_df[(titanic_df['Fare'] >= lower_bound) & (titanic_df['Fare'] <= upper_bound)]
        print(f"Lignes initiales : {initial_count} → après filtrage sur 'Fare' : {titanic_df.shape[0]}")
    except FileNotFoundError:
        print("Fichier 'titanic.csv' introuvable.")

def exercise_standardization_normalization():
    print("\n=== EXERCISE XP 5 : Data Standardization And Normalization ===")
    try:
        titanic_df = pd.read_csv("titanic.csv")
        scaler_std = StandardScaler()
        titanic_df['Fare_Standardized'] = scaler_std.fit_transform(titanic_df[['Fare']])
        scaler_mm = MinMaxScaler()
        titanic_df['Age_Normalized'] = scaler_mm.fit_transform(titanic_df[['Age']])
        print("Aperçu des nouvelles colonnes :")
        print(titanic_df[['Fare', 'Fare_Standardized', 'Age', 'Age_Normalized']].head())
    except FileNotFoundError:
        print("Fichier 'titanic.csv' introuvable.")

def exercise_feature_encoding():
    print("\n=== EXERCISE XP 6 : Feature Encoding ===")
    try:
        titanic_df = pd.read_csv("titanic.csv")
        titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'])
        label_encoder = LabelEncoder()
        titanic_df['Sex'] = label_encoder.fit_transform(titanic_df['Sex'])
        print("Aperçu après encodage :")
        print(titanic_df[['Sex']].head())
    except FileNotFoundError:
        print("Fichier 'titanic.csv' introuvable.")

def exercise_age_transformation():
    print("\n=== EXERCISE XP 7 : Data Transformation For Age Feature ===")
    try:
        titanic_df = pd.read_csv("titanic.csv")
        titanic_df['Age_Group'] = pd.cut(titanic_df['Age'], bins=[0, 12, 18, 60, 100], labels=['Child','Teen','Adult','Senior'])
        titanic_df = pd.get_dummies(titanic_df, columns=['Age_Group'])
        print("Aperçu de la transformation de l'âge :")
        print(titanic_df.head())
    except FileNotFoundError:
        print("Fichier 'titanic.csv' introuvable.")

# -----------------------------------------------------------------------------
# EXERCISES XP GOLD
# -----------------------------------------------------------------------------
def xp_gold_scaling_normalization():
    print("\n=== XP GOLD 1 : Data Scaling And Normalization ===")
    try:
        titanic_df = pd.read_csv("titanic.csv")
        # Standardisation de 'Age' (remplacement des NaN par la moyenne)
        scaler_std = StandardScaler()
        titanic_df['Age_standardized'] = scaler_std.fit_transform(titanic_df[['Age']].fillna(titanic_df['Age'].mean()))
        # Normalisation de 'Fare'
        scaler_mm = MinMaxScaler()
        titanic_df['Fare_normalized'] = scaler_mm.fit_transform(titanic_df[['Fare']])
        print(titanic_df[['Age', 'Age_standardized', 'Fare', 'Fare_normalized']].head())
    except FileNotFoundError:
        print("Fichier 'titanic.csv' introuvable.")

def xp_gold_composite_features():
    print("\n=== XP GOLD 2 : Creating Composite Features ===")
    try:
        titanic_df = pd.read_csv("titanic.csv")
        titanic_df['Family_Size'] = titanic_df['SibSp'] + titanic_df['Parch']
        titanic_df['IsAlone'] = (titanic_df['Family_Size'] == 0).astype(int)
        print(titanic_df[['Family_Size','IsAlone']].head())
    except FileNotFoundError:
        print("Fichier 'titanic.csv' introuvable.")

def xp_gold_data_normalization_visualization():
    print("\n=== XP GOLD 3 : Data Normalization On The Titanic Dataset ===")
    try:
        titanic_df = pd.read_csv("titanic.csv")
        scaler_mm = MinMaxScaler()
        scaler_std = StandardScaler()
        titanic_df['Age_min_max'] = scaler_mm.fit_transform(titanic_df[['Age']].fillna(titanic_df['Age'].mean()))
        titanic_df['Fare_standard'] = scaler_std.fit_transform(titanic_df[['Fare']])
        # Histogrammes
        titanic_df[['Age', 'Age_min_max']].hist()
        plt.suptitle("Age vs Age_min_max")
        plt.show()
        titanic_df[['Fare', 'Fare_standard']].hist()
        plt.suptitle("Fare vs Fare_standard")
        plt.show()
    except FileNotFoundError:
        print("Fichier 'titanic.csv' introuvable.")

def xp_gold_data_reduction_aggregation():
    print("\n=== XP GOLD 4 : Data Reduction And Aggregation ===")
    try:
        titanic_df = pd.read_csv("titanic.csv")
        pca = PCA(n_components=2)
        num_cols = titanic_df.select_dtypes(include=['float64','int64']).dropna().columns
        reduced = pca.fit_transform(titanic_df[num_cols])
        print("Forme des données réduites :", reduced.shape)
    except FileNotFoundError:
        print("Fichier 'titanic.csv' introuvable.")
    except Exception as e:
        print("Erreur lors du PCA :", e)

def xp_gold_normalizing_ecommerce():
    print("\n=== XP GOLD 5 : Normalizing E-Commerce Sales Data ===")
    try:
        superstore_df = pd.read_csv("superstore_sales.csv")
        scaler_mm = MinMaxScaler()
        superstore_df['Sales_normalized'] = scaler_mm.fit_transform(superstore_df[['Sales']])
        superstore_df['Profit_normalized'] = scaler_mm.fit_transform(superstore_df[['Profit']])
        print(superstore_df[['Sales','Sales_normalized','Profit','Profit_normalized']].head())
    except FileNotFoundError:
        print("Fichier 'superstore_sales.csv' introuvable.")

def xp_gold_aggregating_air_quality():
    print("\n=== XP GOLD 6 : Aggregating Air Quality Data ===")
    try:
        air_quality_df = pd.read_csv("air_quality_data.csv")
        air_quality_df['Date'] = pd.to_datetime(air_quality_df['Date'])
        air_quality_grouped = air_quality_df.groupby([air_quality_df['Date'].dt.to_period('M'), 'location']).mean().reset_index()
        print("Premiers agrégats d'Air Quality :")
        print(air_quality_grouped.head())
    except FileNotFoundError:
        print("Fichier 'air_quality_data.csv' introuvable.")

# -----------------------------------------------------------------------------
# EXERCISES XP NINJA
# -----------------------------------------------------------------------------
def xp_ninja_advanced_cleaning():
    print("\n=== XP NINJA 1 : Advanced Data Cleaning And Feature Engineering (NYC Airbnb) ===")
    try:
        airbnb_df = pd.read_csv("nyc_airbnb.csv")
        # Imputation : numérique par médiane, catégoriel par valeur la plus fréquente
        num_imputer = SimpleImputer(strategy='median')
        airbnb_df['reviews_per_month'] = num_imputer.fit_transform(airbnb_df[['reviews_per_month']])
        cat_imputer = SimpleImputer(strategy='most_frequent')
        airbnb_df['last_review'] = cat_imputer.fit_transform(airbnb_df[['last_review']])
        # Détection d’outliers sur 'price'
        Q1 = airbnb_df['price'].quantile(0.25)
        Q3 = airbnb_df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        airbnb_df = airbnb_df[(airbnb_df['price'] >= lower_bound) & (airbnb_df['price'] <= upper_bound)]
        # Création de nouvelles features
        airbnb_df['booking_rate'] = airbnb_df['number_of_reviews'] / airbnb_df['availability_365']
        airbnb_df['price_per_person'] = airbnb_df['price'] / airbnb_df['accommodates']
        airbnb_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        airbnb_df.fillna(0, inplace=True)
        print("Aperçu du dataset Airbnb après nettoyage et feature engineering :")
        print(airbnb_df.head())
        # Visualisations basiques
        plt.figure(figsize=(10,8))
        sns.heatmap(airbnb_df.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap Airbnb")
        plt.show()
        sns.scatterplot(data=airbnb_df, x='booking_rate', y='price')
        plt.title("Booking Rate vs Price")
        plt.show()
    except FileNotFoundError:
        print("Fichier 'nyc_airbnb.csv' introuvable.")

def xp_ninja_data_integration():
    print("\n=== XP NINJA 2 : Complex Data Integration And Transformation ===")
    try:
        happiness_df = pd.read_csv("world-happiness-report.csv")
        health_df = pd.read_csv("world-health-statistics-2020.csv")
        merged_df = pd.merge(happiness_df, health_df, on='Country')
        scaler = MinMaxScaler()
        merged_df['GDP per Capita Normalized'] = scaler.fit_transform(merged_df[['GDP per Capita']])
        merged_df['Life Expectancy Normalized'] = scaler.fit_transform(merged_df[['Life Expectancy']])
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(merged_df[['GDP per Capita Normalized','Life Expectancy Normalized']])
        pca_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
        # Comparaison graphique avant et après PCA
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.scatter(merged_df['GDP per Capita Normalized'], merged_df['Life Expectancy Normalized'])
        plt.title("Avant PCA")
        plt.subplot(1,2,2)
        plt.scatter(pca_df['PCA1'], pca_df['PCA2'])
        plt.title("Après PCA")
        plt.show()
    except FileNotFoundError as e:
        print("Fichier manquant lors de l'intégration (world-happiness-report.csv ou world-health-statistics-2020.csv).", e)

def xp_ninja_dimensionality_reduction():
    print("\n=== XP NINJA 3 : Exploring Dimensionality Reduction Techniques ===")
    try:
        customers_df = pd.read_csv("customers_dataset.csv")
        # PCA
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(customers_df)
        variance_retained = sum(pca.explained_variance_ratio_)
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(customers_df)
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.scatter(pca_results[:,0], pca_results[:,1])
        plt.title(f"PCA - Variance Retained: {variance_retained:.2f}")
        plt.subplot(1,2,2)
        plt.scatter(tsne_results[:,0], tsne_results[:,1])
        plt.title("t-SNE")
        plt.show()
        print("PCA Analysis:")
        print(" - PCA a conservé {:.2f}% de la variance avec 2 composantes.".format(variance_retained*100))
        print(" - Le graphe PCA montre éventuellement des clusters de données.")
        print("\nt-SNE Analysis:")
        print(" - t-SNE fournit une représentation différente pouvant révéler des structures non linéaires.")
    except FileNotFoundError:
        print("Fichier 'customers_dataset.csv' introuvable.")

# -----------------------------------------------------------------------------
# DAILY CHALLENGE : Data Handling And Analysis In Python
# -----------------------------------------------------------------------------
def daily_challenge():
    print("\n=== DAILY CHALLENGE : Data Handling And Analysis In Python ===")
    try:
        data = pd.read_csv("data_science_job_salary.csv")
        # Normalisation par Min-Max pour 'salary'
        scaler = MinMaxScaler()
        data['normalized_salary'] = scaler.fit_transform(data[['salary']])
        # Réduction de dimension par PCA sur les colonnes numériques
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data.select_dtypes(include=['float64','int64']))
        print("Forme de reduced_data :", reduced_data.shape)
        # Agrégation : moyenne et médiane du salaire par niveau d'expérience
        aggregated = data.groupby('experience_level')['salary'].agg(['mean','median'])
        print("\nAgrégation par experience_level :")
        print(aggregated)
        # Nettoyage simple : renommer des colonnes et imputation forward fill
        data_clean = data.rename(columns={'job_title':'Title', 'job_type':'Type'})
        data_clean.fillna(method='ffill', inplace=True)
        # Visualisation de l'agrégation
        aggregated.plot(kind='bar')
        plt.title("Average and Median Salary by Experience Level")
        plt.ylabel("Salary")
        plt.xlabel("Experience Level")
        plt.show()
    except FileNotFoundError:
        print("Fichier 'data_science_job_salary.csv' introuvable.")
    
# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    green_exercises()
    precipitation_forecasting()
    
    print("\n=== EXERCISES XP ===")
    exercise_duplicate_removal()
    exercise_handling_missing()
    exercise_feature_engineering()
    exercise_outlier_detection()
    exercise_standardization_normalization()
    exercise_feature_encoding()
    exercise_age_transformation()
    
    print("\n=== EXERCISES XP GOLD ===")
    xp_gold_scaling_normalization()
    xp_gold_composite_features()
    xp_gold_data_normalization_visualization()
    xp_gold_data_reduction_aggregation()
    xp_gold_normalizing_ecommerce()
    xp_gold_aggregating_air_quality()
    
    print("\n=== EXERCISES XP NINJA ===")
    xp_ninja_advanced_cleaning()
    xp_ninja_data_integration()
    xp_ninja_dimensionality_reduction()
    
    print("\n=== DAILY CHALLENGE ===")
    daily_challenge()

if __name__ == "__main__":
    main()

