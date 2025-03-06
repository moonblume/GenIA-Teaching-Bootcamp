#!/usr/bin/env python3
"""
Author : Clara Martinez

Mini-project Day – Mobile Price Range Analysis


Description :
  Ce script analyse un dataset de mobiles contenant des caractéristiques numériques (battery_power, ram, etc.)
  et des variables binaires (blue, dual_sim, four_g, etc.) afin d’explorer les facteurs influençant le price_range.
  
Les étapes :
  1. Importation et nettoyage des données (aucune valeur manquante, conversion des types si nécessaire).
  2. Analyse descriptive (statistiques de base, distribution des variables clés).
  3. Analyse de corrélation (mise en évidence des facteurs les mieux corrélés avec price_range).
  4. Visualisations :
       - Histogramme de la distribution de la RAM.
       - Scatter plot de RAM vs price_range.
       - Box plot de battery_power par price_range.
       - Box plot de pixel resolution (px_width) par price_range.
  5. Conclusion des facteurs déterminants pour le price_range (notamment l’importance de la RAM).

© 2025 Octopus. All Rights Reserved.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ---------------------------
# 1. Importation et Nettoyage des Données
# ---------------------------
# Remplacez 'mobile_data.csv' par le chemin vers votre dataset.
data_path = 'mobile_data.csv'
try:
    df = pd.read_csv(data_path, encoding='utf-8')
except Exception as e:
    print("Erreur lors du chargement du dataset :", e)
    exit()

print("Aperçu du dataset :")
print(df.head())
print("\nInformations sur le dataset :")
df.info()

# Vérification des valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# Le dataset ne comporte pas de valeurs manquantes – sinon on pourrait utiliser df.fillna() ou dropna().

# La plupart des colonnes catégoriques (blue, dual_sim, four_g, three_g, touch_screen, wifi) sont déjà au format numérique.

# ---------------------------
# 2. Analyse Descriptive et Nettoyage
# ---------------------------
print("\nStatistiques Descriptives :")
print(df.describe())

# Exemple d’observation :
# Les variables battery_power, ram, px_width, and px_height montrent une large variabilité.
# Les variables fc et pc (caméras) ont des valeurs relativement faibles en moyenne.

# ---------------------------
# 3. Analyse de Corrélation
# ---------------------------
corr_matrix = df.corr()
print("\nMatrice de Corrélation :")
print(corr_matrix)

# Visualisation de la heatmap de la corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Parmi les corrélations, notez que 'ram' montre une très forte corrélation positive avec price_range.
# D’autres variables modérément corrélées : battery_power, px_width et px_height.

# ---------------------------
# 4. Visualisations
# ---------------------------
# Histogramme de la distribution de la RAM
plt.figure(figsize=(10, 6))
sns.histplot(df['ram'], bins=30, kde=True, color='skyblue')
plt.title("Distribution de la RAM")
plt.xlabel("RAM (en MB)")
plt.ylabel("Fréquence")
plt.show()

# Scatter plot RAM vs Price Range
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ram', y='price_range', data=df, color='darkblue', alpha=0.7)
plt.title("Relation entre RAM et Price Range")
plt.xlabel("RAM (MB)")
plt.ylabel("Price Range")
plt.show()

# Box plot de battery_power par price_range
plt.figure(figsize=(10, 6))
sns.boxplot(x='price_range', y='battery_power', data=df, palette='Pastel1')
plt.title("Répartition de Battery Power selon le Price Range")
plt.xlabel("Price Range")
plt.ylabel("Battery Power")
plt.show()

# Box plot de px_width (résolution horizontale) par price_range
plt.figure(figsize=(10, 6))
sns.boxplot(x='price_range', y='px_width', data=df, palette='Paired')
plt.title("Répartition de Pixel Resolution (px_width) selon le Price Range")
plt.xlabel("Price Range")
plt.ylabel("Pixel Width")
plt.show()


