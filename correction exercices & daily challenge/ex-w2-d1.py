
"""
Author : Clara Martinez 

Script d'introduction à la Data Analysis et à la catégorisation des types de données.
Ce script réalise les exercices suivants :

Exercice 1 : Rédaction d'un rapport introductif sur la data analysis.
Exercice 2 : Chargement des datasets et affichage des premières lignes.
Exercice 3 : Identification des types de colonnes (qualitatif vs quantitatif).
Exercice 4 : Exploration du dataset Iris et classification de ses colonnes.
Exercice 5 : Calcul de mesures statistiques (moyenne, médiane, mode) et création d'un histogramme.
Exercice 6 : Observation et sélection de colonnes intéressantes dans le dataset "How Much Sleep Do Americans Really Get?".

Avant d’exécuter ce script, vérifie que tu as installé les bibliothèques pandas, matplotlib et seaborn.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mode, StatisticsError

# ─────────────────────────────
def exercice1_rapport():
    """
    Exercice 1 – Rapport introductif
    Rédige un court rapport qui définit la data analysis, son importance, et trois domaines d’application.
    Le rapport est affiché dans la console.
    """
    rapport = """
    Rapport - Introduction à la Data Analysis

    1. Définition de la Data Analysis :
       La data analysis consiste à inspecter, nettoyer, transformer et modéliser des données afin
       d’en extraire des informations utiles pour soutenir la prise de décision et identifier des tendances.

    2. Importance de la Data Analysis en contexte moderne :
       Face à l'abondance des données, la capacité à analyser et interpréter ces dernières permet
       de prendre des décisions éclairées dans des domaines variés (finance, santé, marketing, etc.).

    3. Domaines d’application :
       - Finance : Prédiction des tendances du marché, gestion des risques et détection de fraudes.
       - Santé : Analyse des effets thérapeutiques, optimisation des soins et recherche médicale.
       - Marketing : Segmentation de la clientèle, analyse de campagnes publicitaires et suivi comportemental.
    """
    print(rapport)

# ─────────────────────────────
def exercice2_load_and_describe():
    """
    Exercice 2 – Chargement et description initiale des datasets
    Charge les fichiers CSV "sleep.csv" et "mental_health.csv".
    Pour chaque dataset, affiche les premières lignes ainsi qu'une brève description.
    """
    datasets = {
        "How Much Sleep Do Americans Really Get?": "sleep.csv",
        "Global Trends in Mental Health Disorder and Credit Card Approvals": "mental_health.csv"
    }

    for nom, fichier in datasets.items():
        print("────────────────────────────────────────")
        print("Chargement du dataset :", nom, f"({fichier})")
        try:
            df = pd.read_csv(fichier)
            print("\nPremières lignes :")
            print(df.head(), "\n")
            print("Description du dataset :")
            print(df.info(), "\n")
            print("Résumé statistique (colonnes numériques) :")
            print(df.describe(), "\n")
        except FileNotFoundError:
            print(f"Erreur : Le fichier '{fichier}' est introuvable. Veuillez vérifier son emplacement.\n")

# ─────────────────────────────
def exercice3_identify_data_types():
    """
    Exercice 3 – Identification des types de colonnes
    Pour chacun des datasets chargés précédemment, catégorise chaque colonne comme quantitative (numérique)
    ou qualitative (catégorique/texte) et explique brièvement le choix.
    """
    datasets = {
        "How Much Sleep Do Americans Really Get?": "sleep.csv",
        "Global Trends in Mental Health Disorder and Credit Card Approvals": "mental_health.csv"
    }

    for nom, fichier in datasets.items():
        print("────────────────────────────────────────")
        print("Analyse des types de colonnes pour :", nom)
        try:
            df = pd.read_csv(fichier)
            for col in df.columns:
                dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    type_col = "Quantitative"
                    raison = "Les données sont numériques et peuvent être utilisées pour des calculs statistiques."
                else:
                    type_col = "Qualitative"
                    raison = "Les données sont de type objet/catégorique et décrivent des attributs non numériques."
                print(f"Colonne: {col} -> {type_col} | Raison : {raison}")
            print("\n")
        except FileNotFoundError:
            print(f"Erreur : Le fichier '{fichier}' est introuvable.\n")

# ─────────────────────────────
def exercice4_explore_iris():
    """
    Exercice 4 – Exploration du dataset Iris
    Charge le dataset Iris via Seaborn, affiche les premières lignes et identifie les colonnes quantitatives
    et qualitatives.
    """
    print("────────────────────────────────────────")
    print("Chargement et exploration du dataset Iris")
    iris = sns.load_dataset('iris')
    print("\nPremières lignes du dataset Iris:")
    print(iris.head(), "\n")
    
    print("Classification des colonnes du dataset Iris :")
    for col in iris.columns:
        if col == "species":
            print(f"{col} -> Qualitative | Raison : Représente l'espèce de la fleur (catégorie).")
        else:
            print(f"{col} -> Quantitative | Raison : Mesure numérique permettant des calculs statistiques.")
    print("\n")

# ─────────────────────────────
def exercice5_basic_analysis():
    """
    Exercice 5 – Analyse de base : calcul de moyennes, médiane, mode et création d'un histogramme.
    Utilise le dataset Iris pour analyser la colonne "sepal_length".
    """
    print("────────────────────────────────────────")
    print("Analyse statistique de la colonne 'sepal_length' du dataset Iris")
    iris = sns.load_dataset('iris')
    col = "sepal_length"
    data = iris[col]
    
    mean_val = data.mean()
    median_val = data.median()
    try:
        mode_val = mode(data)
    except StatisticsError:
        mode_val = "Aucun mode unique ne peut être identifié"
    
    print(f"\nMoyenne : {mean_val:.2f}")
    print(f"Médiane : {median_val:.2f}")
    print(f"Mode : {mode_val}")
    
    # Création d'un histogramme
    plt.figure(figsize=(8,5))
    plt.hist(data, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Sepal Length")
    plt.ylabel("Fréquence")
    plt.title("Histogramme de 'sepal_length'")
    plt.grid(True)
    plt.show()

# ─────────────────────────────
def exercice6_observation_skills():
    """
    Exercice 6 – Observation et sélection de colonnes dans le dataset "How Much Sleep Do Americans Really Get?"
    Charge le dataset, affiche la liste des colonnes et identifie celles qui sont numériques et potentiellement intéressantes
    pour des analyses quantitatives.
    """
    print("────────────────────────────────────────")
    print("Observation dans le dataset 'How Much Sleep Do Americans Really Get?'")
    try:
        df = pd.read_csv("sleep.csv")
        print("\nListe des colonnes disponibles :")
        print(df.columns.tolist(), "\n")
        colonnes_interessantes = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                colonnes_interessantes.append(col)
        print("Colonnes numériques identifiées (pour une analyse quantitative) :")
        print(colonnes_interessantes)
        print("\nExemple d'analyse : comparer 'SleepDuration' (si présente) en fonction d'autres variables, pour observer des tendances ou groupes démographiques.\n")
    except FileNotFoundError:
        print("Erreur : Le fichier 'sleep.csv' est introuvable. Veuillez le placer dans le même répertoire que ce script.\n")

# ─────────────────────────────
def main():
    print("\nExercice 1 : Rapport introductif")
    exercice1_rapport()
    
    print("\nExercice 2 : Chargement et description initiale des datasets")
    exercice2_load_and_describe()
    
    print("\nExercice 3 : Identification des types de colonnes")
    exercice3_identify_data_types()
    
    print("\nExercice 4 : Exploration du dataset Iris")
    exercice4_explore_iris()
    
    print("\nExercice 5 : Analyse de base avec calculs statistiques et graphique")
    exercice5_basic_analysis()
    
    print("\nExercice 6 : Observation et sélection de colonnes d'intérêt")
    exercice6_observation_skills()

if __name__ == "__main__":
    main()
