
#!/usr/bin/env python3
"""
Author : Clara Martinez

NumPy Advanced Solution Exercises 

Ce script comporte 4 sections :

────────────────────────
• SOLUTIONS EXERCISES XP  
  ◦ Ex.1: Opérations matricielles sur une matrice 3x3 (calcul du déterminant et de l'inverse)
  ◦ Ex.2: Analyse statistique sur 50 nombres aléatoires (mean, median, std)
  ◦ Ex.3: Manipulation de dates – création d’un array de dates pour janvier 2023 et conversion de format
  ◦ Ex.4: Manipulation de données avec NumPy et Pandas (DataFrame, sélection conditionnelle, agrégation)
  ◦ Ex.5: Représentation d’image – création d’une image grayscale 5x5 à partir d’un array
  ◦ Ex.6: Test d’hypothèse basique avec un test t apparié (utilisation de scipy.stats)
  ◦ Ex.7: Comparaison complexe d’arrays : comparaison élément par élément
  ◦ Ex.8: Manipulation de séries temporelles – génération d’une série pour 2023 puis découpage
  ◦ Ex.9: Conversion de données – conversion d’un array NumPy en DataFrame puis retour en array
  ◦ Ex.10: Visualisation basique – tracé d’un graphe linéaire de données aléatoires

• SOLUTIONS EXERCISES XP GOLD  
  ◦ Ex.1: Opérations matricielles avancées – matrice 5x5, calcul des valeurs propres et normalisation
  ◦ Ex.2: Analyse de distribution statistique – création d’un dataset normal et histogramme
  ◦ Ex.3: Prévision en série temporelle – régression linéaire sur données mensuelles
  ◦ Ex.4: Agrégation de données avec Pandas – group by et calcul de moyennes et sommes
  ◦ Ex.5: Visualisation complexe – création d’une figure à 4 sous-graphiques (line, scatter, bar)

• SOLUTIONS EXERCISES XP NINJA  
  ◦ Ex.1: Opérations sur matrice diagonale – extraire la diagonale et reconstruire une matrice diagonale
  ◦ Ex.2: Opérations conditionnelles sur array – remplacement d’éléments > 0.5 par -1
  ◦ Ex.3: Normalisation de données – normaliser un array en utilisant StandardScaler
  ◦ Ex.4: Calcul de coefficients de corrélation entre 2 arrays aléatoires
  ◦ Ex.5: Tracage d’une ligne de tendance sur une série temporelle à l’aide de np.polyfit

• DAILY CHALLENGE – Global Power Plant Database  
  ◦ Data Import & Cleaning (lecture du CSV, imputation de valeurs manquantes)
  ◦ Exploratory Data Analysis (résumé statistiques, matrice de corrélation)
  ◦ Analyse statistique (exemples d’opérations NumPy)
  ◦ Visualisation (exemples de graphiques avec Seaborn et Matplotlib)

Assurez-vous d’avoir installé les packages suivants (si ce n’est pas déjà fait) :
    pip install numpy pandas matplotlib seaborn scipy scikit-learn
────────────────────────────────────────

"""

# IMPORTS GLOBAUX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import scipy.linalg as la

# =============================================================================
# SOLUTIONS EXERCISES XP
# =============================================================================
def xp_exercises():
    print("\n--- SOLUTIONS EXERCISES XP ---\n")
    # Exercise 1: Matrix Operations – déterminant et inverse d'une matrice 3x3
    def calculate_determinant(matrix):
        return np.linalg.det(matrix)
    def calculate_inverse(matrix):
        return np.linalg.inv(matrix)
    matrix_3x3 = np.array([[1, 2, 3],
                           [0, 1, 4],
                           [5, 6, 0]])
    det = calculate_determinant(matrix_3x3)
    inv = calculate_inverse(matrix_3x3)
    print("Ex1 - Matrice 3x3:\n", matrix_3x3)
    print("Déterminant:", det)
    print("Inverse:\n", inv, "\n")
    
    # Exercise 2: Statistical Analysis over 50 random numbers
    data = np.random.rand(50)
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    print("Ex2 - 50 nombres aléatoires:\n", data)
    print("Mean:", mean, "Median:", median, "Standard Deviation:", std_dev, "\n")
    
    # Exercise 3: Date Manipulation for January 2023
    dates = np.array([
      '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
      '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
      '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15',
      '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20',
      '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25',
      '2023-01-26', '2023-01-27', '2023-01-28', '2023-01-29', '2023-01-30',
      '2023-01-31'
    ], dtype='datetime64')
    # Convertir le format 'YYYY-MM-DD' en 'YYYY/MM/DD'
    converted_dates = np.array(dates, dtype='datetime64[D]').astype(str)
    converted_dates = np.char.replace(converted_dates, '-', '/')
    print("Ex3 - Dates converties:\n", converted_dates, "\n")
    
    # Exercise 4: Data Manipulation with NumPy and Pandas
    df_rand = pd.DataFrame(np.random.rand(10, 5), columns=['A','B','C','D','E'])
    selected_data = df_rand[df_rand['A'] > 0.5]
    sum_data = df_rand.sum()
    average_data = df_rand.mean()
    print("Ex4 - DataFrame aléatoire:\n", df_rand)
    print("Sélection conditionnelle (A > 0.5):\n", selected_data)
    print("Somme par colonne:\n", sum_data)
    print("Moyenne par colonne:\n", average_data, "\n")
    
    # Exercise 5: Image Representation – 5x5 grayscale image
    image = np.random.randint(0, 256, (5, 5))
    plt.imshow(image, cmap='gray')
    plt.title('Ex5 - 5x5 Grayscale Image')
    plt.show()
    
    # Exercise 6: Basic Hypothesis Testing with paired t-test
    prod_before = np.random.normal(loc=50, scale=10, size=30)
    prod_after = prod_before + np.random.normal(loc=5, scale=3, size=30)
    t_stat, p_value = stats.ttest_rel(prod_after, prod_before)
    print("Ex6 - Test t apparié:")
    print("t-statistic:", t_stat, "p-value:", p_value)
    if p_value < 0.05:
        print("Hypothèse supportée : amélioration significative.\n")
    else:
        print("Hypothèse non supportée : pas d'amélioration significative.\n")
    
    # Exercise 7: Complex Array Comparison: comparer deux arrays
    array1 = np.random.randint(1, 10, size=5)
    array2 = np.random.randint(1, 10, size=5)
    comparison = array1 > array2
    print("Ex7 - Array1:", array1)
    print("Array2:", array2)
    print("Comparaison (Array1 > Array2):", comparison, "\n")
    
    # Exercise 8: Time Series Data Manipulation with Pandas
    date_range = pd.date_range(start='2023-01-01', end='2023-12-31')
    time_series = pd.Series(np.arange(len(date_range)), index=date_range)
    jan_mar = time_series['2023-01-01':'2023-03-31']
    apr_jun = time_series['2023-04-01':'2023-06-30']
    jul_sep = time_series['2023-07-01':'2023-09-30']
    oct_dec = time_series['2023-10-01':'2023-12-31']
    print("Ex8 - Time Series Slicing:")
    print("Jan-Mar:\n", jan_mar.head())
    print("Apr-Jun:\n", apr_jun.head())
    print("Jul-Sep:\n", jul_sep.head())
    print("Oct-Dec:\n", oct_dec.head(), "\n")
    
    # Exercise 9: Data Conversion between NumPy array and Pandas DataFrame
    np_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df_from_np = pd.DataFrame(np_array, columns=['Column1','Column2','Column3'])
    converted_np = df_from_np.to_numpy()
    print("Ex9 - NumPy Array:\n", np_array)
    print("Converti en DataFrame:\n", df_from_np)
    print("Converti de nouveau en Array:\n", converted_np, "\n")
    
    # Exercise 10: Basic Visualization – Line graph of random numbers
    data_vis = np.random.rand(10)
    plt.plot(data_vis)
    plt.title("Ex10 - Random Number Line Graph")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()


# =============================================================================
# SOLUTIONS EXERCISES XP GOLD
# =============================================================================
def xp_gold_exercises():
    print("\n--- SOLUTIONS EXERCISES XP GOLD ---\n")
    # Exercise 1: Advanced Matrix Operations – 5x5 random matrix, eigenvalues/vectors & normalization
    np.random.seed(0)
    matrix_5x5 = np.random.rand(5, 5)
    eigenvalues, eigenvectors = la.eig(matrix_5x5)
    normalized_matrix = matrix_5x5 / np.linalg.norm(matrix_5x5)
    print("XP Gold Ex1 - Matrice 5x5:\n", matrix_5x5)
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
    print("Normalized Matrix:\n", normalized_matrix, "\n")
    
    # Exercise 2: Statistical Distribution Analysis – normal distribution and histogram
    np.random.seed(0)
    normal_data = np.random.normal(loc=0.0, scale=1.0, size=1000)
    plt.figure(figsize=(8,6))
    plt.hist(normal_data, bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Normally Distributed Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    # Exercise 3: Time Series Forecasting – Linear regression on monthly sales data
    np.random.seed(0)
    monthly_sales = np.random.randint(100, 500, size=12)
    months = np.arange(1, 13).reshape(-1, 1)
    sales = monthly_sales.reshape(-1, 1)
    model = LinearRegression()
    model.fit(months, sales)
    next_month = np.array([[13]])
    forecast = model.predict(next_month)
    print("XP Gold Ex3 - Monthly Sales:", monthly_sales)
    print("Forecast for month 13:", forecast[0][0], "\n")
    
    # Exercise 4: Pandas Data Aggregation
    data_dict = {
        'Product': ['Product A', 'Product B', 'Product C', 'Product A', 'Product B'],
        'Sales': [200, 150, 300, 250, 180],
        'Month': ['January', 'February', 'March', 'April', 'May']
    }
    df_sample = pd.DataFrame(data_dict)
    grouped = df_sample.groupby('Product').agg({'Sales': ['mean', 'sum']})
    print("XP Gold Ex4 - Grouped and Aggregated Data:\n", grouped, "\n")
    
    # Exercise 5: Complex Data Visualization – multi-plot layout (line, scatter, bar)
    np.random.seed(0)
    x = np.linspace(0, 10, 100)
    y_line = 2 * x + 1
    y_scatter = np.random.rand(100) * 100
    y_bar = np.random.randint(1, 100, 10)
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    # Line plot
    axes[0,0].plot(x, y_line, color='red')
    axes[0,0].set_title('Line Plot')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,0].grid(True)
    # Scatter plot
    axes[0,1].scatter(x, y_scatter, color='green')
    axes[0,1].set_title('Scatter Plot')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    axes[0,1].grid(True)
    # Bar plot
    axes[1,0].bar(np.arange(1, 11), y_bar, color='blue')
    axes[1,0].set_title('Bar Plot')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    axes[1,0].grid(True)
    # Quatrième subplot vide
    axes[1,1].axis('off')
    plt.suptitle("XP Gold - Multiple Plots")
    plt.show()


# =============================================================================
# SOLUTIONS EXERCISES XP NINJA
# =============================================================================
def xp_ninja_exercises():
    print("\n--- SOLUTIONS EXERCISES XP NINJA ---\n")
    # Exercise 1: Diagonal Matrix Operations – extraire la diagonale et créer une matrice diagonale
    def diagonal_matrix_operations():
        matrix = np.random.rand(4, 4)
        diag_elements = np.diag(matrix)
        diag_matrix = np.diag(diag_elements)
        return diag_matrix
    diag_matrix = diagonal_matrix_operations()
    print("XP Ninja Ex1 - Diagonal Matrix:\n", diag_matrix, "\n")
    
    # Exercise 2: Conditional Array Operations – remplacer éléments > 0.5 par -1
    def conditional_array_operations():
        array = np.random.rand(20)
        array[array > 0.5] = -1
        return array
    cond_array = conditional_array_operations()
    print("XP Ninja Ex2 - Array conditionnelle:\n", cond_array, "\n")
    
    # Exercise 3: Data Normalization using StandardScaler
    def normalize_array():
        array = np.random.rand(20)
        scaler = StandardScaler()
        norm_array = scaler.fit_transform(array.reshape(-1,1)).flatten()
        return norm_array
    norm_array = normalize_array()
    print("XP Ninja Ex3 - Normalized array:\n", norm_array, "\n")
    
    # Exercise 4: Correlation Coefficients between two arrays
    def compute_correlation():
        a = np.random.rand(20)
        b = np.random.rand(20)
        corr = np.corrcoef(a, b)[0,1]
        return corr
    correlation_val = compute_correlation()
    print("XP Ninja Ex4 - Correlation coefficient:", correlation_val, "\n")
    
    # Exercise 5: Time Series Trend Line – ajustement d'une droite à une série temporelle
    def time_series_trend_line():
        days = np.arange(1, 16)
        values = np.random.rand(15)
        coeff = np.polyfit(days, values, 1)
        trend = np.poly1d(coeff)
        plt.plot(days, values, 'o', label='Data')
        plt.plot(days, trend(days), label='Trend Line')
        plt.xlabel('Days')
        plt.ylabel('Values')
        plt.title('Time Series Data with Trend Line')
        plt.legend()
        plt.show()
    time_series_trend_line()


# =============================================================================
# DAILY CHALLENGE – Global Power Plant Database Analysis
# =============================================================================
def daily_challenge():
    print("\n--- DAILY CHALLENGE: Analysis of the Global Power Plant Database ---\n")
    # Note : Adaptez le chemin du fichier CSV selon vos données
    try:
        data = pd.read_csv('/mnt/data/global_power_plant_database.csv')
    except Exception as e:
        print("Erreur de lecture du dataset global_power_plant_database.csv :", e)
        return
    
    # Exploration initiale
    print("Aperçu des 5 premières lignes:\n", data.head())
    # Imputation de valeurs manquantes pour colonnes numériques par la médiane…
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        data[col].fillna(data[col].median(), inplace=True)
    # Pour les colonnes de type objet, remplir avec la modalité la plus fréquente
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Conversion de types si nécessaire (ex. certains champs numériques)
    # data['SomeColumn'] = pd.to_numeric(data['SomeColumn'], errors='coerce')
    
    # EDA
    print("\nRésumé statistique global:\n", data.describe(include='all'))
    corr_matrix = data.corr()
    print("\nMatrice de corrélation:\n", corr_matrix)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Heatmap de la Corrélation")
    plt.show()
    
    # Exemple de visualisation : scatter plot d'une variable numérique (si présente)
    if 'CapacityMW' in data.columns and 'Efficiency' in data.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=data['CapacityMW'], y=data['Efficiency'])
        plt.title("Scatter Plot: CapacityMW vs Efficiency")
        plt.xlabel("CapacityMW")
        plt.ylabel("Efficiency")
        plt.show()
    else:
        print("Les colonnes 'CapacityMW' et 'Efficiency' ne sont pas présentes dans le dataset.")
    
    # Des analyses statistiques supplémentaires avec NumPy
    # Par exemple, calcul de moyenne, médiane, écart-type pour une colonne numérique
    if 'CapacityMW' in data.columns:
        capacity = data['CapacityMW'].to_numpy()
        print("\nStatistiques pour CapacityMW :")
        print("Moyenne:", np.mean(capacity))
        print("Médiane:", np.median(capacity))
        print("Ecart-type:", np.std(capacity))
    
    # Visualisations complémentaires peuvent être ajoutées ici
    print("\nDaily Challenge terminé – n’hésitez pas à compléter votre analyse selon vos besoins.")

# =============================================================================
# MAIN – Exécution de toutes les sections
# =============================================================================
def main():
    xp_exercises()
    xp_gold_exercises()
    xp_ninja_exercises()
    daily_challenge()

if __name__ == "__main__":
    main()
