#!/usr/bin/env python3

"""
Author : Clara Martinez

Scipy and Basic Statistics Solutions Exercises 

Ce script regroupe :

– DC : Data Import, nettoyage, analyse descriptive et visualisation (exemple sur un dataset de crashs).
– XP Exercises : Exemples d’utilisation basique de SciPy
– XP Gold Solutions : Exemples avancés (distributions multivariées, hypothèses, régression)
– XP Ninja Solutions : Analyse sur un dataset ChickWeight (exploration, visualisation et ANOVA)

© 2025 Octopus. All Rights Reserved.
"""

# IMPORTS GLOBAUX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, f_oneway, binom, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import scipy.linalg as la

# =============================================================================
# DC : Daily Challenge – Data Import, Cleaning, Analysis & Visualization
# =============================================================================
def daily_challenge_analysis():
    print("\n=== DAILY CHALLENGE: Data Import, Cleaning & Analysis ===\n")
    # Remplacez 'path_to_dataset.csv' par le chemin vers votre dataset
    url = 'path_to_dataset.csv'  
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print("Erreur lors de la lecture du dataset:", e)
        return

    # Aperçu du dataset
    print("Premières lignes du dataset:\n", df.head())

    # Vérifier les valeurs manquantes
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Traitement des valeurs manquantes
    df.dropna(inplace=True)  # Remplacez par fillna() si nécessaire

    # Conversion des dates (supposons que le dataset comporte une colonne 'date')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Vérification de la cohérence des données catégorielles (exemple sur 'region')
    if 'region' in df.columns:
        print("\nValeurs uniques dans la colonne 'region':")
        print(df['region'].unique())
        # Standardisation (strip et lowercase)
        df['region'] = df['region'].str.strip().str.lower()

    # Statistiques de base
    print("\nBasic Statistics:")
    print(df.describe())

    # Exemple d’analyse : nombre de crashs et total de fatalities (supposant colonnes 'crash_id' et 'fatalities')
    if 'crash_id' in df.columns and 'fatalities' in df.columns:
        total_crashes = df['crash_id'].nunique()
        total_fatalities = df['fatalities'].sum()
        print(f"\nTotal number of crashes: {total_crashes}")
        print(f"Total fatalities: {total_fatalities}")
    else:
        print("\nColonnes 'crash_id' ou 'fatalities' introuvables.")

    # Création d'une colonne 'survival_rate' si 'passengers' existe
    if 'passengers' in df.columns and 'fatalities' in df.columns:
        df['survival_rate'] = 1 - df['fatalities'] / df['passengers']
        print("\nAverage Survival Rate:")
        print(df['survival_rate'].mean())

    # Analyse temporelle : extraire l'année et grouper
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        crashes_per_year = df.groupby('year').size()
        plt.figure(figsize=(10,6))
        crashes_per_year.plot(kind='line')
        plt.title("Airplane Crashes Over Time")
        plt.xlabel("Year")
        plt.ylabel("Number of Crashes")
        plt.grid(True)
        plt.show()
    
    # Calcul de statistiques clés sur 'fatalities', si disponible
    if 'fatalities' in df.columns:
        fatalities_mean = df['fatalities'].mean()
        fatalities_median = df['fatalities'].median()
        fatalities_std = df['fatalities'].std()
        print(f"\nFatalities - Mean: {fatalities_mean}, Median: {fatalities_median}, Std Dev: {fatalities_std}")

        # Hypothèse: comparaison des moyennes entre deux décennies (ex. 1980-1990 vs 2000-2010)
        if 'year' in df.columns:
            df['decade'] = (df['year'] // 10) * 10
            decade_80s_90s = df[df['decade'].isin([1980, 1990])]['fatalities']
            decade_00s_10s = df[df['decade'].isin([2000, 2010])]['fatalities']
            if len(decade_80s_90s) > 0 and len(decade_00s_10s) > 0:
                t_stat, p_value = stats.ttest_ind(decade_80s_90s, decade_00s_10s)
                print(f"\nT-test Results - t-statistic: {t_stat}, p-value: {p_value}")
            else:
                print("\nPas suffisamment de données pour comparer les décennies.")

    # Visualisation : Histogramme des fatalities
    if 'fatalities' in df.columns:
        plt.figure(figsize=(10,6))
        sns.histplot(df['fatalities'], bins=30, kde=True)
        plt.title("Distribution of Fatalities in Airplane Crashes")
        plt.xlabel("Fatalities")
        plt.ylabel("Frequency")
        plt.show()
    
    # Visualisation : Bar plot des crashs par region (si 'region' disponible)
    if 'region' in df.columns:
        plt.figure(figsize=(10,6))
        sns.countplot(x='region', data=df)
        plt.title("Number of Crashes by Region")
        plt.xlabel("Region")
        plt.ylabel("Number of Crashes")
        plt.xticks(rotation=45)
        plt.show()


# =============================================================================
# XP EXERCISES – Basic Usage of SciPy and Descriptive Statistics
# =============================================================================
def xp_exercises_scipy():
    print("\n=== XP EXERCISES with SciPy ===\n")
    import scipy
    print("SciPy Version:", scipy.__version__)

    # Exercise 2: Descriptive Statistics
    data_list = [12, 15, 13, 12, 18, 20, 22, 21]
    t_mean = stats.tmean(data_list)
    t_median = stats.tmedian(data_list)
    t_variance = stats.tvar(data_list)
    t_std = stats.tstd(data_list)
    print("\nExercise 2: Descriptive Statistics")
    print("Mean:", t_mean)
    print("Median:", t_median)
    print("Variance:", t_variance)
    print("Standard Deviation:", t_std)

    # Exercise 3: Understanding Distributions – Normal Distribution
    mean_val = 50
    std_val = 10
    size = 1000
    norm_data = np.random.normal(mean_val, std_val, size)
    plt.figure(figsize=(8,6))
    plt.hist(norm_data, bins=30, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_val, std_val)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title("Normal Distribution (Mean=50, Std=10)")
    plt.show()

    # Exercise 4: T-Test Application
    d1 = np.random.normal(50, 10, 100)
    d2 = np.random.normal(60, 10, 100)
    t_stat, p_val = stats.ttest_ind(d1, d2)
    print("\nExercise 4: T-Test Application")
    print("T-statistic:", t_stat)
    print("P-value:", p_val)

    # Exercise 5: Linear Regression Analysis on house data
    house_sizes = np.array([50, 70, 80, 100, 120])
    house_prices = np.array([150000, 200000, 210000, 250000, 280000])
    reg_result = stats.linregress(house_sizes, house_prices)
    print("\nExercise 5: Linear Regression Analysis")
    print("Slope:", reg_result.slope)
    print("Intercept:", reg_result.intercept)
    predicted_price = reg_result.slope * 90 + reg_result.intercept
    print("Predicted price for a 90m² house:", predicted_price)

    # Exercise 6: Understanding ANOVA
    fert1 = [5, 6, 7, 6, 5]
    fert2 = [7, 8, 7, 9, 8]
    fert3 = [4, 5, 4, 3, 4]
    f_stat, p_anova = f_oneway(fert1, fert2, fert3)
    print("\nExercise 6: ANOVA")
    print("F-statistic:", f_stat)
    print("P-value:", p_anova)

    # Exercise 7: Probability Distributions – Binomial
    n, p = 10, 0.5
    prob_5_heads = binom.pmf(5, n, p)
    print("\nExercise 7: Binomial Distribution")
    print("Probability of exactly 5 heads in 10 flips:", prob_5_heads)

    # Exercise 8: Correlation Coefficients
    df_corr = pd.DataFrame({'age': [23, 25, 30, 35, 40],
                            'income': [35000, 40000, 50000, 60000, 70000]})
    pearson_corr, _ = pearsonr(df_corr['age'], df_corr['income'])
    spearman_corr, _ = spearmanr(df_corr['age'], df_corr['income'])
    print("\nExercise 8: Correlation Coefficients")
    print("Pearson correlation coefficient:", pearson_corr)
    print("Spearman correlation coefficient:", spearman_corr)


# =============================================================================
# XP GOLD SOLUTIONS – Advanced Statistical Analysis & Regression
# =============================================================================
def xp_gold_solutions():
    print("\n=== XP GOLD SOLUTIONS ===\n")
    # Exercise 1: Multivariate Normal Distribution
    univariate_data = norm.rvs(size=1000)
    mean_vec = [0, 0]
    cov_matrix = [[1, 0], [0, 1]]
    multivariate_data = stats.multivariate_normal.rvs(mean=mean_vec, cov=cov_matrix, size=1000)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.hist(univariate_data, bins=30, density=True, alpha=0.6, color='blue')
    plt.title("Univariate Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.subplot(1,2,2)
    plt.scatter(multivariate_data[:,0], multivariate_data[:,1], alpha=0.6, color='red')
    plt.title("Multivariate Normal Distribution")
    plt.xlabel("X value")
    plt.ylabel("Y value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Exercise 2: Advanced Probability Distributions
    # Exemple de Poisson : Probabilité d'obtenir 3 événements si λ = 5
    lambda_val = 5
    poisson_prob = stats.poisson.pmf(3, lambda_val)
    # Exemple d’Exponential : densité de probabilité à x=2 avec lambda=1/3
    exp_val = stats.expon.pdf(2, scale=3)
    print("\nXP Gold Ex2:")
    print("Poisson probability (k=3, λ=5):", poisson_prob)
    print("Exponential PDF (x=2, scale=3):", exp_val)

    # Exercise 3: Statistical Hypothesis Testing – ANOVA across three samples (simulation)
    np.random.seed(0)
    sample1 = np.random.normal(20000, 3000, 30)
    sample2 = np.random.normal(22000, 3500, 30)
    sample3 = np.random.normal(25000, 5000, 30)
    anova_result = f_oneway(sample1, sample2, sample3)
    print("\nXP Gold Ex3: ANOVA Test")
    print("F-statistic:", anova_result.statistic, "p-value:", anova_result.pvalue)

    # Exercise 4: Linear Regression Analysis on sample data
    np.random.seed(0)
    X_vals = np.random.rand(100) * 50    # Exemple: heures étudiées
    Y_vals = 2.5 * X_vals + np.random.randn(100) * 10  # Exemple: scores de test
    linreg_result = stats.linregress(X_vals, Y_vals)
    print("\nXP Gold Ex4: Linear Regression")
    print("Slope:", linreg_result.slope)
    print("Intercept:", linreg_result.intercept)
    predicted = linreg_result.slope * 40 + linreg_result.intercept
    print("Predicted value for X = 40:", predicted)


# =============================================================================
# XP NINJA SOLUTIONS – Analysis on the ChickWeight Dataset
# =============================================================================
def xp_ninja_solutions():
    print("\n=== XP NINJA SOLUTIONS ===\n")
    # Charger le dataset ChickWeight (assurez-vous qu'il est accessible via 'ChickWeight.csv')
    try:
        chick_weights = pd.read_csv('ChickWeight.csv')
    except Exception as e:
        print("Erreur lors de la lecture de ChickWeight.csv:", e)
        return
    
    # Task 1: Data Exploration
    print("Premières lignes de ChickWeight:\n", chick_weights.head())

    # Task 2: Visualisation - scatter plot poids vs âge par régime
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=chick_weights, x='age', y='weight', hue='diet')
    plt.title("Weight vs Age of Chicks on Different Diets")
    plt.xlabel("Age (weeks)")
    plt.ylabel("Weight (grams)")
    plt.legend(title='Diet')
    plt.grid(True)
    plt.show()

    # Task 3: ANOVA test sur les poids par régime
    weights_by_diet = [group['weight'].values for name, group in chick_weights.groupby('diet')]
    anova_result = f_oneway(*weights_by_diet)
    print("\nANOVA Test Result on chick weights by diet:", anova_result)
    
    # Task 4: Growth Analysis – Calcul de la moyenne par régime et âge
    avg_growth = chick_weights.groupby(['diet','age']).mean().reset_index()
    print("\nAverage weight gain per week by diet (extrait):\n", avg_growth.head())


# =============================================================================
# MAIN – Exécution de toutes les sections
# =============================================================================
def main():
    print("\n=== SCI PY & STATISTICS SOLUTIONS ===")
    daily_challenge_analysis()
    xp_exercises_scipy()
    xp_gold_solutions()
    xp_ninja_solutions()

if __name__ == "__main__":
    main()
