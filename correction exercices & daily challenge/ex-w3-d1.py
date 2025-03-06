#!/usr/bin/env python3
"""
Author : Clara Martinez 

Solutions Exercises -  Intro to NumPy

Ce script présente :

• EXERCISES XP : Exemples basiques (création d'arrays, extraction, inversion, opérations, etc.)
• EXERCISES XP GOLD : Quelques opérations avancées (normalisation, multiplication matricielle, extraction d'entiers)
• EXERCISES XP NINJA : Exemples encore plus poussés (remplacement de valeurs, valeurs communes, tri, rang, etc.)
• DAILY CHALLENGE : Préparation de données (températures aléatoires pour 10 villes sur 12 mois), analyse et visualisation

Assurez-vous d'avoir installé NumPy et Pandas, ainsi que Matplotlib pour le Daily Challenge.
 pip install numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# EXERCISES XP – SOLUTIONS BASIQUES
# =============================================================================

def xp_exercise_1():
    # Exercise 1: Créer un array 1D contenant les nombres de 0 à 9.
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print("XP Exercise 1:\n", arr)
    return arr

def xp_exercise_2():
    # Exercise 2: Convertir la liste [3.14, 2.17, 0, 1, 2] en array NumPy et convertir en entier.
    arr = np.array([3.14, 2.17, 0, 1, 2], dtype=int)
    print("XP Exercise 2:\n", arr)
    return arr

def xp_exercise_3():
    # Exercise 3: Créer un array 3x3 avec les valeurs de 1 à 9.
    arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print("XP Exercise 3:\n", arr)
    return arr

def xp_exercise_4():
    # Exercise 4: Créer un array 2D de forme (4,5) rempli de nombres aléatoires.
    arr = np.random.rand(4, 5)
    print("XP Exercise 4:\n", arr)
    return arr

def xp_exercise_5(array):
    # Exercise 5: Sélectionner la deuxième ligne d'un array 2D.
    second_row = array[1, :]
    print("XP Exercise 5:\n", second_row)
    return second_row

def xp_exercise_6(array):
    # Exercise 6: Inverser l'ordre des éléments d'un array 1D.
    reversed_arr = array[::-1]
    print("XP Exercise 6:\n", reversed_arr)
    return reversed_arr

def xp_exercise_7():
    # Exercise 7: Créer une matrice identité 4x4.
    ident = np.eye(4)
    print("XP Exercise 7:\n", ident)
    return ident

def xp_exercise_8(array):
    # Exercise 8: Calculer la somme et la moyenne d'un array 1D.
    total = array.sum()
    avg = array.mean()
    print("XP Exercise 8:\n Sum =", total, " Mean =", avg)
    return total, avg

def xp_exercise_9():
    # Exercise 9: Créer un array avec les éléments de 1 à 20 et le remodeler en une matrice 4x5.
    arr = np.arange(1, 21).reshape(4, 5)
    print("XP Exercise 9:\n", arr)
    return arr

def xp_exercise_10(array):
    # Exercise 10: Extraire tous les nombres impairs d'un array.
    odds = array[array % 2 == 1]
    print("XP Exercise 10:\n", odds)
    return odds

# =============================================================================
# EXERCISES XP GOLD – OPÉRATIONS AVANCÉES
# =============================================================================

def xp_gold_exercise_1():
    # XP Gold Exercise 1: Créer une matrice 5x5 avec des valeurs aléatoires et en extraire les min et max.
    matrix = np.random.rand(5, 5)
    min_value = matrix.min()
    max_value = matrix.max()
    print("XP Gold Exercise 1:\nMatrice:\n", matrix)
    print("Min =", min_value, "Max =", max_value)
    return matrix, min_value, max_value

def xp_gold_exercise_2():
    # XP Gold Exercise 2: Normaliser une matrice 3x3 aléatoire.
    matrix = np.random.rand(3, 3)
    normalized_matrix = (matrix - matrix.mean()) / matrix.std()
    print("XP Gold Exercise 2:\nMatrice:\n", matrix)
    print("Matrice normalisée:\n", normalized_matrix)
    return matrix, normalized_matrix

def xp_gold_exercise_3():
    # XP Gold Exercise 3: Créer un array 1D de 50 éléments équidistants entre 0 et 10, exclus.
    evenly_spaced_array = np.linspace(0, 10, 50, endpoint=False)
    print("XP Gold Exercise 3:\n", evenly_spaced_array)
    return evenly_spaced_array

def xp_gold_exercise_4():
    # XP Gold Exercise 4: Multiplier une matrice 5x3 par une matrice 3x2.
    matrix_a = np.random.rand(5, 3)
    matrix_b = np.random.rand(3, 2)
    product = np.dot(matrix_a, matrix_b)
    print("XP Gold Exercise 4:\nMatrice A:\n", matrix_a)
    print("Matrice B:\n", matrix_b)
    print("Produit:\n", product)
    return matrix_a, matrix_b, product

def xp_gold_exercise_5():
    # XP Gold Exercise 5: Extraire la partie entière d'un array aléatoire en 5 méthodes.
    random_array = np.random.rand(5) * 10
    method_1 = np.floor(random_array)
    method_2 = np.ceil(random_array) - 1
    method_3 = random_array.astype(int)
    method_4 = np.trunc(random_array)
    method_5 = np.array([int(i) for i in random_array])
    print("XP Gold Exercise 5:\nArray:", random_array)
    print("Méthode 1 (floor):", method_1)
    print("Méthode 2 (ceil-1):", method_2)
    print("Méthode 3 (astype int):", method_3)
    print("Méthode 4 (trunc):", method_4)
    print("Méthode 5 (list comprehension):", method_5)
    return random_array, method_1, method_2, method_3, method_4, method_5

# =============================================================================
# EXERCISES XP NINJA – OPÉRATIONS AVANCÉES / MANIPULATIONS COMPLEXES
# =============================================================================

def xp_ninja_exercise_1():
    # XP Ninja Exercise 1: Créer une matrice 5x5 aléatoire et remplacer la valeur maximum par 0.
    matrix = np.random.rand(5, 5)
    print("XP Ninja Exercise 1:\nMatrice avant modification:\n", matrix)
    matrix[matrix == matrix.max()] = 0
    print("Matrice après modification (max remplacé par 0):\n", matrix)
    return matrix

def xp_ninja_exercise_2():
    # XP Ninja Exercise 2: Trouver les valeurs communes entre deux arrays aléatoires de taille 5.
    array1 = np.random.randint(0, 10, 5)
    array2 = np.random.randint(0, 10, 5)
    common_values = np.intersect1d(array1, array2)
    print("XP Ninja Exercise 2:\nArray 1:", array1, "\nArray 2:", array2)
    print("Valeurs communes:", common_values)
    return array1, array2, common_values

def xp_ninja_exercise_3():
    # XP Ninja Exercise 3: Créer un array 1D aléatoire de taille 10 et le trier en ordre ascendant puis descendant.
    array = np.random.rand(10)
    ascending = np.sort(array)
    descending = ascending[::-1]
    print("XP Ninja Exercise 3:\nArray original:", array)
    print("Tri ascendant:", ascending)
    print("Tri descendant:", descending)
    return array, ascending, descending

def xp_ninja_exercise_4():
    # XP Ninja Exercise 4: Générer une matrice 4x4 aléatoire et en trouver le rang.
    matrix = np.random.rand(4, 4)
    rank = np.linalg.matrix_rank(matrix)
    print("XP Ninja Exercise 4:\nMatrice:\n", matrix)
    print("Rang de la matrice:", rank)
    return matrix, rank

def xp_ninja_exercise_5():
    # XP Ninja Exercise 5: Créer un array 2D avec 1 sur la bordure et 0 à l'intérieur.
    border_array = np.ones((5, 5))
    border_array[1:-1, 1:-1] = 0
    print("XP Ninja Exercise 5:\n", border_array)
    return border_array

# =============================================================================
# DAILY CHALLENGE – DATA PREPARATION, ANALYSIS & VISUALIZATION
# =============================================================================

def daily_challenge():
    # Data Preparation
    np.random.seed(0)  # Pour reproductibilité
    temperatures = np.random.uniform(-5, 35, (10, 12))  # 10 villes, 12 mois
    cities = ["City " + str(i) for i in range(1, 11)]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    temp_df = pd.DataFrame(temperatures, index=cities, columns=months)
    print("Daily Challenge – Data Preparation")
    print(temp_df.head())
    
    # Data Analysis – Moyenne annuelle et identification de la ville la plus chaude et la plus froide
    annual_avg_temp = temp_df.mean(axis=1)
    highest_avg_temp_city = annual_avg_temp.idxmax()
    lowest_avg_temp_city = annual_avg_temp.idxmin()
    print("Ville avec la température annuelle moyenne la plus élevée :", highest_avg_temp_city)
    print("Ville avec la température annuelle moyenne la plus basse :", lowest_avg_temp_city)
    
    # Data Visualization – Courbes mensuelles pour chaque ville
    plt.figure(figsize=(12, 6))
    for city in cities:
        if city in [highest_avg_temp_city, lowest_avg_temp_city]:
            plt.plot(months, temp_df.loc[city], label=city, linewidth=2.5, linestyle='--')
        else:
            plt.plot(months, temp_df.loc[city], label=city, alpha=0.7)
    plt.title("Monthly Temperature Trends for Each City")
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# MAIN – Appel de toutes les fonctions d'exemples
# =============================================================================

def main():
    print("\n--- EXERCISES XP ---")
    arr1 = xp_exercise_1()
    xp_exercise_2()
    arr2 = xp_exercise_3()
    xp_exercise_4()
    # Pour exercise 5 et 6, utilisons arr2 de l'exercice 3
    xp_exercise_5(arr2)
    xp_exercise_6(arr1)
    xp_exercise_7()
    xp_exercise_8(arr1)
    xp_exercise_9()
    xp_exercise_10(arr1)
    
    print("\n--- EXERCISES XP GOLD ---")
    xp_gold_exercise_1()
    xp_gold_exercise_2()
    xp_gold_exercise_3()
    xp_gold_exercise_4()
    xp_gold_exercise_5()
    
    print("\n--- EXERCISES XP NINJA ---")
    xp_ninja_exercise_1()
    xp_ninja_exercise_2()
    xp_ninja_exercise_3()
    xp_ninja_exercise_4()
    xp_ninja_exercise_5()
    
    print("\n--- DAILY CHALLENGE ---")
    daily_challenge()

if __name__ == "__main__":
    main()

