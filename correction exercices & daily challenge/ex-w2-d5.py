
#!/usr/bin/env python3
"""
Author : Clara Martinez 

Mini-project Solution : Data Analysis For Marketing Strategy

Objectifs :
  • Analyse par zones géographiques (pays, états, villes)
  • Analyse par clients (scattering des ventes et profits, identification de clients « outstanding »)
  • Analyse par catégories et sous-catégories
  • Analyse en séries temporelles (par année, par mois)
  • Daily Challenge interactif avec Plotly et Plotnine

Ce script utilise le dataset "Sample - Superstore.csv" disponible, par exemple, sur Kaggle.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotnine import *

import datetime
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# MAIN ANALYSIS – Data Import & Preprocessing
# ---------------------------
def main():
    # Import du dataset (adapté à ton chemin)
    df = pd.read_csv('/kaggle/input/superstore-dataset-final/Sample - Superstore.csv', encoding='ISO-8859-1')
    print("Aperçu du dataset:")
    print(df.head())
    print("\nInformations sur le dataset:")
    df.info()

    # Conversion des dates 'Order Date' et 'Ship Date'
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y')

    # Création de nouvelles variables temporelles
    df['OrderY'] = df['Order Date'].dt.year
    df['OrderM'] = df['Order Date'].dt.month
    df['OrderD'] = df['Order Date'].dt.day

    # Calcul de la rentabilité (Profitability)
    df['Profitability'] = df['Profit'] / df['Sales']

    # ---------------------------
    # Analyse par zones géographiques
    # ---------------------------
    print("\nVentes par pays :")
    countries_sales = df.groupby('Country')['Sales'].sum().sort_values(ascending=False)
    print(countries_sales)
    countries_sales.plot.barh(title="Ventes par pays")
    plt.show()

    # La totalité des ventes se situe aux États-Unis. Analyse par états :
    colors = ['blue'] * 18 + ['red'] * 2
    top20_state_sales = df.groupby('State')['Sales'].sum().sort_values(ascending=True).tail(20)
    top20_state_sales.plot.barh(color=colors, title="Top 20 états par ventes")
    plt.show()

    top20_state_profit = df.groupby('State')['Profit'].sum().sort_values(ascending=True).tail(20)
    top20_state_profit.plot.barh(color=colors, title="Top 20 états par profit")
    plt.show()

    # ---------------------------
    # Analyse par clients – Cas de California et New York (états stratégiques)
    # ---------------------------
    df_cal = df[df['State']=='California']
    df_new = df[df['State']=='New York']

    # Pour la Californie
    df_cal_customer = df_cal.groupby('Customer Name')['Sales'].sum().to_frame()
    df_cal_customer['Profit'] = df_cal.groupby('Customer Name')['Profit'].sum()
    df_cal_customer['balckorred'] = df_cal_customer['Profit'].apply(lambda x: 'red' if x < 0 else 'black')
    sns.scatterplot(data=df_cal_customer, hue='balckorred', x='Sales', y='Profit')
    plt.xlim(0, 14000)
    plt.ylim(-1000, 5000)
    plt.title("Ventes et Profit par clients en Californie")
    plt.show()

    # Pour New York
    df_new_customer = df_new.groupby('Customer Name')['Sales'].sum().to_frame()
    df_new_customer['Profit'] = df_new.groupby('Customer Name')['Profit'].sum()
    df_new_customer['balckorred'] = df_new_customer['Profit'].apply(lambda x: 'red' if x < 0 else 'black')
    sns.scatterplot(data=df_new_customer, hue='balckorred', x='Sales', y='Profit')
    plt.xlim(0, 14000)
    plt.ylim(-1000, 5000)
    plt.title("Ventes et Profit par clients à New York")
    plt.show()

    print("\nClients remarquables à New York (triés par ventes) :")
    print(df_new_customer.sort_values(by=['Sales'], ascending=False).head())

    # Exclure 'Tom Ashbrook' pour observer l'impact sur les moyennes
    df_new_noTom = df_new_customer.drop(index='Tom Ashbrook', errors='ignore')
    print("\nStatistiques clients à New York sans 'Tom Ashbrook' :")
    print(df_new_noTom.describe().T)

    # Différences de rentabilité par états
    df_state = df.groupby('State')['Sales'].sum().to_frame()
    df_state['Profit'] = df.groupby('State')['Profit'].sum()
    df_state['balckorred'] = df_state['Profit'].apply(lambda x: 'red' if x < 0 else 'black')
    sns.scatterplot(data=df_state, hue='balckorred', x='Sales', y='Profit')
    plt.title("Ventes vs Profit par état")
    plt.show()

    # ---------------------------
    # Analyse par villes
    # ---------------------------
    top20_city_sales = df.groupby('City')['Sales'].sum().sort_values(ascending=True).tail(20)
    top20_city_sales.plot.barh(color=colors, title="Top 20 villes par ventes")
    plt.show()

    top20_city_profit = df.groupby('City')['Profit'].sum().sort_values(ascending=True).tail(20)
    top20_city_profit.plot.barh(color=colors, title="Top 20 villes par profit")
    plt.show()

    df_city = df.groupby('City')['Sales'].sum().to_frame()
    df_city['Profit'] = df.groupby('City')['Profit'].sum()
    df_city['balckorred'] = df_city['Profit'].apply(lambda x: 'red' if x < 0 else 'black')
    sns.scatterplot(data=df_city, hue='balckorred', x='Sales', y='Profit')
    plt.title("Ventes vs Profit par ville")
    plt.show()

    # ---------------------------
    # Analyse par clients – Vue générale et Pareto
    # ---------------------------
    top20_customer_sales = df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=True).tail(20)
    top20_customer_sales.plot.barh(color=colors, title="Top 20 clients par ventes")
    plt.show()

    plt.figure(figsize=(12,10))
    df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).cumsum().plot(title="Courbe cumulative des ventes")
    plt.show()

    df1 = pd.DataFrame(df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False))
    threshold = df1.quantile(0.7, interpolation='higher')['Sales']
    pareto_sales = df1[df1['Sales'] >= threshold].sum() / df['Sales'].sum()
    print("\nAnalyse Pareto pour les ventes : Top 30% des clients représentent {:.2%} des ventes".format(pareto_sales['Sales']))

    top20_customer_profit = df.groupby('Customer Name')['Profit'].sum().sort_values(ascending=True).tail(20)
    top20_customer_profit.plot.barh(color=colors, title="Top 20 clients par profit")
    plt.show()

    plt.figure(figsize=(12,10))
    df.groupby('Customer Name')['Profit'].sum().sort_values(ascending=False).cumsum().plot(title="Courbe cumulative du profit")
    plt.show()

    df2 = pd.DataFrame(df.groupby('Customer Name')['Profit'].sum().sort_values(ascending=False))
    threshold2 = df2.quantile(0.7, interpolation='higher')['Profit']
    pareto_profit = df2[df2['Profit'] >= threshold2].sum() / df['Profit'].sum()
    print("\nAnalyse Pareto pour le profit : Top 30% des clients représentent {:.2%} du profit".format(pareto_profit['Profit']))

    df_customer = df.groupby('Customer Name')['Sales'].sum().to_frame()
    df_customer['Profit'] = df.groupby('Customer Name')['Profit'].sum()
    df_customer['balckorred'] = df_customer['Profit'].apply(lambda x: 'red' if x < 0 else 'black')
    sns.scatterplot(data=df_customer, hue='balckorred', x='Sales', y='Profit')
    plt.title("Ventes vs Profit par client")
    plt.show()

    # ---------------------------
    # Analyse par catégories de produits
    # ---------------------------
    df.groupby('Category')['Sales'].sum().sort_values(ascending=True).plot.barh(title="Ventes par catégorie")
    plt.show()

    df.groupby('Category')['Profit'].sum().sort_values(ascending=True).plot.barh(title="Profit par catégorie")
    plt.show()

    df_category = df.groupby('Category')['Sales'].sum().to_frame()
    df_category['Profit'] = df.groupby('Category')['Profit'].sum()
    df_category.plot.scatter(x='Sales', y='Profit', title="Ventes vs Profit par catégorie")
    plt.show()

    colors_sub = ['red', 'red', 'blue', 'blue','blue','blue','blue','blue','yellow','yellow',
                  'yellow', 'yellow', 'yellow', 'yellow','yellow','yellow','yellow']
    df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).plot.bar(color=colors_sub, title="Ventes par sous-catégorie")
    plt.show()

    df.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False).plot.bar(color=colors_sub, title="Profit par sous-catégorie")
    plt.show()

    df_subcategory = df.groupby('Sub-Category')['Sales'].sum().to_frame()
    df_subcategory['Profit'] = df.groupby('Sub-Category')['Profit'].sum()
    df_subcategory['balckorred'] = df_subcategory['Profit'].apply(lambda x: 'red' if x < 0 else 'black')
    sns.scatterplot(data=df_subcategory, hue='balckorred', x='Sales', y='Profit')
    plt.title("Ventes vs Profit par sous-catégorie")
    plt.show()

    # ---------------------------
    # Analyse de séries temporelles
    # ---------------------------
    df.groupby('OrderY')['Sales'].sum().plot.barh(title="Ventes par année")
    plt.show()
    df.groupby('OrderY')['Profit'].sum().plot.barh(title="Profit par année")
    plt.show()

    plt.figure(figsize=(20,5))
    df.groupby(['OrderY','OrderM'])['Sales'].sum().plot(title="Tendance mensuelle des ventes")
    plt.show()

    plt.figure(figsize=(20,5))
    df.groupby(['OrderY','OrderM'])['Profit'].sum().plot(title="Tendance mensuelle du profit")
    plt.show()

    print("\nConclusion : Prioriser états, villes, clients et catégories pour augmenter les ventes et améliorer la rentabilité. La prévision via des analyses de séries temporelles peut être envisagée.")

    # -----------------------------------------------------
    # Daily Challenge : Visualisation interactive avec Plotly et Plotnine
    # -----------------------------------------------------
    interactive_daily_challenge()

# ---------------------------
# DAILY CHALLENGE – Visualisations interactives
# ---------------------------
def interactive_daily_challenge():
    print("\n=== DAILY CHALLENGE : Interactive Data Visualization ===")
    # Pour cet exemple, nous simulons un dataset
    data = pd.DataFrame({
        'Year': [2017, 2018, 2019, 2020, 2021],
        'Sales': [15000, 18000, 22000, 21000, 25000],
        'Country': ['USA', 'USA', 'USA', 'USA', 'USA'],
        'Product': ['A', 'B', 'A', 'C', 'B'],
        'Discount': [5, 10, 7, 12, 9],
        'Profit': [2000, 2500, 3000, 2800, 3500]
    })

    # Graphique interactif (line chart) avec Plotly
    fig = px.line(data, x='Year', y='Sales', title='Tendance des ventes par année')
    fig.show()

    # Carte interactive (dummy) avec Plotly
    fig_geo = px.scatter_geo(data, locations='Country', color='Sales',
                             hover_name='Country', size='Sales',
                             projection='natural earth', title='Répartition des ventes par pays')
    fig_geo.show()

    # Diagramme en barres avec Plotnine
    bar_chart = (ggplot(data, aes(x='Product', y='Sales')) +
                 geom_bar(stat='identity') +
                 theme(axis_text_x=element_text(rotation=90, hjust=1)) +
                 ggtitle('Top 10 produits par ventes'))
    print(bar_chart)

    # Scatter plot avec Plotnine
    scatter_plot = (ggplot(data, aes(x='Discount', y='Profit')) +
                    geom_point() +
                    ggtitle('Relation entre profit et remise'))
    print(scatter_plot)

if __name__ == "__main__":
    main()

