
"""
Author : Clara Martinez

Ce script contient :

  – LESSON 1 : Exemples de tracés classiques (line plot, bar plot, histogram, box plot, scatter plot, treemap)
  – LESSON 2 : Graphiques complémentaires (line plot avec style, bar chart, histogram, pie chart)
  – LESSON 3 : Visualisations avec Seaborn (histplot, pairplot, heatmap, facet grid)
  – DAILY CHALLENGE : Analyse du World Happiness Report avec tracés multiples
  – XP EXERCISES : Exemples simples (line plot, bar plot, histogram, countplot, scatter plot)
  – XP GOLD EXERCISES : Graphiques avancés (bar graph, filtrage hiérarchique, histplot, box plot, pairplot)
  – XP NINJA EXERCISES : Graphiques annotés et configuration dynamique

Avant d’exécuter ce script, vérifie que tous les fichiers requis sont présents et que les packages sont installés.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import numpy as np
import pandas as pd
import re

# ---------------------------
# LESSON 1 : Graphiques classiques
# ---------------------------
def lesson1():
    print("\n=== LESSON 1 ===")
    # Line Plot
    x = range(1, 8)
    y = [72, 74, 76, 80, 82, 78, 75]
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title("Line Plot: Weekly Temperature")
    plt.xlabel("Day")
    plt.ylabel("Temperature (°F)")
    plt.show()

    # Bar Plot
    categories = ['Apples', 'Bananas', 'Cherries', 'Dates']
    sales = [100, 150, 200, 90]
    plt.figure()
    plt.bar(categories, sales, color='skyblue')
    plt.title("Bar Plot: Fruit Sales")
    plt.xlabel("Fruit")
    plt.ylabel("Sales")
    plt.show()

    # Histogram
    data = np.random.normal(loc=0, scale=1, size=1000)
    plt.figure()
    plt.hist(data, bins=20, edgecolor='black')
    plt.title("Histogram: Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    # Box Plot
    data_box = np.random.rand(50) * 100
    plt.figure()
    plt.boxplot(data_box)
    plt.title("Box Plot: Random Data")
    plt.ylabel("Values")
    plt.show()

    # Scatter Plot
    x_scatter = np.random.rand(50) * 100
    y_scatter = np.random.rand(50) * 100
    plt.figure()
    plt.scatter(x_scatter, y_scatter, c='blue', alpha=0.5)
    plt.title("Scatter Plot: Random Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

    # Treemap
    sizes = [40, 30, 20, 10]
    labels = ['A', 'B', 'C', 'D']
    plt.figure()
    squarify.plot(sizes=sizes, label=labels, alpha=0.8)
    plt.title("Treemap Example")
    plt.axis('off')
    plt.show()

# ---------------------------
# LESSON 2 : Graphiques complémentaires
# ---------------------------
def lesson2():
    print("\n=== LESSON 2 ===")
    # Line Plot for y = x^2
    x = range(0, 11)
    y = [i**2 for i in x]
    plt.figure()
    plt.plot(x, y, color='green', linestyle='--', marker='o')
    plt.title("Line Plot: y = x^2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()

    # Bar Chart for Sales
    products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
    sales = [200, 400, 300, 450, 500]
    plt.figure()
    plt.bar(products, sales, color=['red', 'blue', 'green', 'orange', 'purple'])
    plt.title("Bar Chart: Product Sales")
    plt.xlabel("Products")
    plt.ylabel("Sales")
    plt.show()

    # Histogram for Normal Distribution
    data = np.random.normal(0, 1, 500)
    plt.figure()
    plt.hist(data, bins=15, color='cyan', edgecolor='black')
    plt.title("Histogram: Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    # Pie Chart for Daily Activities
    activities = ['Work', 'Sleep', 'Exercise', 'Leisure']
    time_spent = [8, 7, 2, 7]
    explode = (0.1, 0, 0, 0)
    plt.figure()
    plt.pie(time_spent, labels=activities, autopct='%1.1f%%', explode=explode, shadow=True)
    plt.title("Pie Chart: Daily Activities")
    plt.show()

# ---------------------------
# LESSON 3 : Visualisations avec Seaborn
# ---------------------------
def lesson3():
    print("\n=== LESSON 3 ===")
    # Histogram with Seaborn using the 'tips' dataset
    tips = sns.load_dataset('tips')
    plt.figure()
    sns.histplot(tips['total_bill'], bins=15, kde=True, color='blue')
    plt.title("Histogram: Total Bill")
    plt.show()

    # Pair Plot with Iris Dataset
    iris = sns.load_dataset('iris')
    sns.pairplot(iris, hue='species', markers=['o', 's', 'D'])
    plt.suptitle("Pair Plot: Iris Dataset", y=1.02)
    plt.show()

    # Heatmap for Correlation Matrix of Iris
    correlation = iris.corr()
    plt.figure()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("Heatmap: Correlation Matrix")
    plt.show()

    # Facet Grid for Titanic Dataset (histogram de 'age' par sexe et classe)
    titanic = sns.load_dataset('titanic')
    g = sns.FacetGrid(titanic, col="sex", row="class", margin_titles=True)
    g.map(sns.histplot, "age", kde=True)
    plt.show()

# ---------------------------
# DAILY CHALLENGE : World Happiness Report with Matplotlib
# ---------------------------
def daily_challenge():
    print("\n=== DAILY CHALLENGE ===")
    # Remplacer 'path_to_dataset' par le répertoire où se trouve le fichier, ici "2019.csv"
    dataset_path = "2019.csv"
    try:
        df = pd.read_csv(dataset_path)
        print("Dataset Preview:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
    except Exception as e:
        print("Erreur lors du chargement du dataset :", e)
        return

    # Vérifier les valeurs manquantes
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Suppression des lignes avec des valeurs manquantes
    df = df.dropna()

    # Scatter plot pour "Social support" vs "Happiness Score"
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Social support'], df['Score'], alpha=0.7, c='blue', edgecolors='k')
    plt.title("Social Support vs. Happiness Score (2019)", fontsize=14)
    plt.xlabel("Social Support", fontsize=12)
    plt.ylabel("Happiness Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Agrégation : Moyenne de "GDP per capita" et "Healthy life expectancy" par "Region"
    # On suppose qu'une colonne 'Region' existe – sinon il faudra la créer
    if 'Region' in df.columns:
        region_group = df.groupby('Region')[['GDP per capita', 'Healthy life expectancy']].mean().reset_index()

        # Bar et line plot combinés
        fig, ax1 = plt.subplots(figsize=(10, 6))
        bar_width = 0.4
        bar_positions = np.arange(len(region_group['Region']))
        ax1.bar(bar_positions, region_group['GDP per capita'], color='orange', width=bar_width, label='GDP per Capita')
        ax1.set_ylabel("GDP per Capita", fontsize=12)
        ax1.set_xticks(bar_positions)
        ax1.set_xticklabels(region_group['Region'], rotation=45, ha='right', fontsize=10)
        ax1.set_title("GDP per Capita and Healthy Life Expectancy by Region (2019)", fontsize=14)

        ax2 = ax1.twinx()
        ax2.plot(bar_positions, region_group['Healthy life expectancy'], color='green', marker='o', label='Healthy Life Expectancy')
        ax2.set_ylabel("Healthy Life Expectancy (Years)", fontsize=12)
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        plt.tight_layout()
        plt.show()
    else:
        print("La colonne 'Region' n'existe pas dans le dataset.")

# ---------------------------
# XP EXERCISES : Exemples simples de visualisation
# ---------------------------
def xp_exercises():
    print("\n=== XP EXERCISES ===")
    # Exercise 1: Compréhension de la visualisation
    print("La Data Visualization permet de rendre les données plus compréhensibles et de mettre en évidence les tendances.")

    # Exercise 2: Line Plot pour variations de température sur une semaine
    temperature = [72, 74, 76, 80, 82, 78, 75]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    plt.figure()
    plt.plot(days, temperature, marker='o')
    plt.xlabel("Day")
    plt.ylabel("Temperature (°F)")
    plt.title("Temperature Variation Over a Week")
    plt.grid()
    plt.show()

    # Exercise 3: Bar Chart pour ventes mensuelles
    months = ["January", "February", "March", "April", "May"]
    sales = [5000, 5500, 6200, 7000, 7500]
    plt.figure()
    plt.bar(months, sales, color='skyblue')
    plt.xlabel("Month")
    plt.ylabel("Sales Amount ($)")
    plt.title("Monthly Sales Data")
    plt.show()

    # Exercise 4: Histogram pour la distribution du CGPA
    data_cgpa = {'CGPA': [3.5, 3.7, 3.8, 3.9, 4.0, 3.6, 3.4, 3.8, 3.5, 3.9]}
    df_cgpa = pd.DataFrame(data_cgpa)
    plt.figure()
    sns.histplot(df_cgpa['CGPA'], kde=True, color='green')
    plt.title("Distribution of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.show()

    # Exercise 5: Countplot pour comparer le niveau d'anxiété par genre
    data_anxiety = {
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Anxiety': ['Yes', 'No', 'Yes', 'Yes', 'No']
    }
    df_anxiety = pd.DataFrame(data_anxiety)
    plt.figure()
    sns.countplot(x='Gender', hue='Anxiety', data=df_anxiety, palette='Set2')
    plt.title("Anxiety Levels Across Genders")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.show()

    # Exercise 6: Scatter Plot pour l'âge et la présence d'attaques de panique
    data_panic = {'Age': [20, 22, 24, 26, 28, 30, 32],
                  'Panic Attack': [1, 0, 1, 1, 0, 0, 1]}
    df_panic = pd.DataFrame(data_panic)
    plt.figure()
    sns.scatterplot(x='Age', y='Panic Attack', data=df_panic)
    plt.title("Age vs Panic Attack Occurrence")
    plt.xlabel("Age")
    plt.ylabel("Panic Attack (Yes=1, No=0)")
    plt.show()

# ---------------------------
# XP GOLD EXERCISES : Graphiques avancés
# ---------------------------
def xp_gold_exercises():
    print("\n=== XP GOLD EXERCISES ===")
    # Exercise 1: Bar Graph
    categories = ["Electronics", "Clothing", "Home Goods", "Books"]
    sales = [20000, 15000, 18000, 12000]
    plt.figure()
    plt.bar(categories, sales, color='orange')
    plt.title("Sales by Product Category")
    plt.xlabel("Product Categories")
    plt.ylabel("Sales Amount ($)")
    plt.show()

    # Exercise 2: Hierarchical Indexing
    multi_index_data = {
        ('Canada', 'Toronto', '2023-01-01'): 32,
        ('Canada', 'Vancouver', '2023-01-02'): 35,
        ('Canada', 'Montreal', '2023-01-03'): 28,
    }
    index = pd.MultiIndex.from_tuples(multi_index_data.keys(), names=['Country', 'City', 'Date'])
    temp_df = pd.DataFrame(list(multi_index_data.values()), index=index, columns=['Temperature'])
    print("Hierarchical Indexing - filtered data:")
    filtered_data = temp_df.loc[("Canada", slice(None), slice("2023-01-01", "2023-01-03"))]
    filtered_data = filtered_data[filtered_data['Temperature'] > 30]
    print(filtered_data)

    # Exercise 3: Advanced Filtering with Hierarchical Indices
    data_salary = {
        ('Dept1', 'Alice'): 60000,
        ('Dept1', 'Bob'): 45000,
        ('Dept2', 'Charlie'): 55000,
        ('Dept2', 'David'): 50000
    }
    index_salary = pd.MultiIndex.from_tuples(data_salary.keys(), names=['Department', 'Employee'])
    salary_df = pd.DataFrame(list(data_salary.values()), index=index_salary, columns=['Salary'])
    filtered_salary = salary_df[salary_df['Salary'] > 50000]
    print("\nFiltered Salary (Salary > 50000):")
    print(filtered_salary)

    # Exercise 4: Visualizing Movie Durations
    movie_data = {'Movie Duration': [120, 135, 110, 140, 150, 125]}
    df_movie = pd.DataFrame(movie_data)
    plt.figure()
    sns.histplot(df_movie['Movie Duration'], kde=True, color='purple')
    plt.title("Distribution of Movie Durations")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Frequency")
    plt.show()

    # Exercise 5: Box Plot for Audience and Tomato Meter Scores
    score_data = {
        'Audience Score': [80, 85, 78, 92, 88],
        'Tomato Meter': [75, 80, 70, 90, 85]
    }
    df_score = pd.DataFrame(score_data)
    melted_df = df_score.melt(var_name="Score Type", value_name="Score")
    plt.figure()
    sns.boxplot(x='Score Type', y='Score', data=melted_df, palette='coolwarm')
    plt.title("Audience vs Tomato Meter Scores")
    plt.show()

    # Exercise 6: Pair Plot of Financial Data
    financial_data = {
        'Production Budget': [100, 200, 150, 250, 300],
        'Opening Weekend': [50, 100, 80, 130, 160],
        'Domestic Box Office': [120, 250, 200, 300, 400],
        'Worldwide Box Office': [200, 400, 350, 500, 600]
    }
    df_financial = pd.DataFrame(financial_data)
    sns.pairplot(df_financial)
    plt.suptitle("Pair Plot: Financial Data", y=1.02)
    plt.show()

# ---------------------------
# XP NINJA EXERCISES : Graphiques annotés et dynamiques
# ---------------------------
def xp_ninja_exercises():
    print("\n=== XP NINJA EXERCISES ===")
    # Exercise 1: Annotated Line Graph for Temperature Records
    temperature = [32, 35, 30, 40, 28, 38, 42]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    plt.figure()
    plt.plot(days, temperature, marker='o')
    plt.annotate("Extreme Heat", xy=(6, 42), xytext=(4, 44),
                 arrowprops=dict(facecolor='red', arrowstyle='->'))
    plt.xlabel("Day")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Fluctuations")
    plt.grid()
    plt.show()

    # Exercise 2: (Référence à XP Gold Exercise 2) – Affichage hiérarchique déjà présenté
    print("XP NINJA Exercise 2 : Voir XP GOLD Exercise 2 pour l'exemple de filtrage hiérarchique.")

    # Exercise 3: Dynamic Subplot Configuration (Exemple simple sans interaction)
    # Par exemple, afficher deux subplots côte à côte
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(days, temperature, marker='o', color='magenta')
    plt.title("Dynamic Subplot 1")
    plt.subplot(1,2,2)
    plt.bar(days, temperature, color='cyan')
    plt.title("Dynamic Subplot 2")
    plt.tight_layout()
    plt.show()
    
    print("XP NINJA Exercise 3 : Exemple de configuration dynamique de subplots exécuté.")

# ---------------------------
# MAIN
# ---------------------------
def main():
    lesson1()
    lesson2()
    lesson3()
    daily_challenge()
    xp_exercises()
    xp_gold_exercises()
    xp_ninja_exercises()
    
if __name__ == "__main__":
    main()

