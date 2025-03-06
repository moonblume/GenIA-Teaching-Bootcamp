"""

Author : Clara Martinez

• Daily Challenge : Transformation et visualisation d'images issues d'un dataset Kaggle.
• Import / Export de données (CSV, Excel, JSON) et lecture de données depuis une URL.
• Exercices XP sur la distinction entre données structurées et non structurées ainsi que divers cas d'analyse.

Avant d’exécuter ce script, assure-toi d’avoir installé les bibliothèques suivantes :
pip install kaggle tensorflow matplotlib pandas opencv-python pillow requests faker textblob

Si tu utilises Google Colab, certaines commandes (upload de fichiers, configuration de kaggle)
peuvent être dé-commentées.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import rotate
import cv2
from PIL import ImageEnhance, Image
import requests
import re

Pour XP Ninja (analyse sentimentale)
from textblob import TextBlob

Pour XP Gold création de catalogue fictif
from faker import Faker

#------------------------------------------------------------------------------

SECTION 1 : TRANSFORMATION ET VISUALISATION D'IMAGES
#------------------------------------------------------------------------------
def rotate_image_30_degrees(image):
"""Retourne l'image tournée de 30°."""
return rotate(image, 30, reshape=False, mode='nearest')

def vertical_flip(image):
"""Retourne l'image retournée verticalement (rotation de 180° avec reshape=False)."""
# Ici, nous utilisons une rotation de 180°.
return rotate(image, 180, reshape=False, mode='nearest')

def flip_image(image, mode='horizontal'):
"""Retourne l'image retournée selon le mode spécifié (horizontal ou vertical)."""
if mode == 'horizontal':
flipped_image = cv2.flip(image, 1)
elif mode == 'vertical':
flipped_image = cv2.flip(image, 0)
else:
raise ValueError("Invalid mode. Mode must be 'horizontal' or 'vertical'.")
return flipped_image

def adjust_contrast(image):
"""Ajuste le contraste de l'image en doublant le contraste."""
# On part d'un array d'image à valeurs entre 0 et 1.
pil_img = Image.fromarray((image * 255).astype(np.uint8))
contrast = ImageEnhance.Contrast(pil_img)
pil_img_enhanced = contrast.enhance(2.0)
return np.array(pil_img_enhanced) / 255.0

def image_transformation_and_visualization():
"""
Télécharge (via kaggle si besoin) le dataset "flower-color-images", extrait quelques images,
et applique plusieurs transformations : flip horizontal, rotation de 30°, flip vertical, et ajustement du contraste.
"""
#––– Pour Google Colab, décommente si nécessaire :
# from google.colab import files
# files.upload()
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d olgabelitskaya/flower-color-images
# !unzip flower-color-images.zip


Collapse
# Récupérer 5 images aléatoires depuis le dossier "flowers/flowers"
img_folder = os.path.join('flowers', 'flowers')
if not os.path.isdir(img_folder):
    print("Le dossier 'flowers/flowers' est introuvable. Vérifie que le dataset a été décompressé correctement.")
    return

list_img = []
plt.figure(figsize=(20, 20))
files_list = os.listdir(img_folder)
for i in range(5):
    file = random.choice(files_list)
    image_path = os.path.join(img_folder, file)
    try:
        img = mpimg.imread(image_path)
        list_img.append(img)
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {file}: {e}")
# Affichage des images d'origine
for i, img in enumerate(list_img):
    ax = plt.subplot(3, 5, i + 1)
    ax.set_title(f"Origine:\n{file}")
    plt.imshow(img)
    plt.axis('off')
plt.suptitle("Images d'origine", fontsize=16)
plt.show()

# Transformation 1 : Flip horizontal avec cv2
horizontal_flip_list = []
plt.figure(figsize=(20, 20))
for img in list_img:
    horizontal_flip_list.append(flip_image(img, mode='horizontal'))
for i, img in enumerate(horizontal_flip_list):
    ax = plt.subplot(3, 5, i + 1)
    ax.set_title("Flip Horizontal")
    plt.imshow(img)
    plt.axis('off')
plt.show()

# Transformation 2 : Rotation de 30 degrés
rotated_list = []
plt.figure(figsize=(20, 20))
for img in list_img:
    rotated_list.append(rotate_image_30_degrees(img))
for i, img in enumerate(rotated_list):
    ax = plt.subplot(3, 5, i + 1)
    ax.set_title("Rotation 30°")
    plt.imshow(img)
    plt.axis('off')
plt.show()

# Transformation 3 : Flip vertical (rotation à 180°)
vertical_flip_list = []
plt.figure(figsize=(20, 20))
for img in list_img:
    vertical_flip_list.append(vertical_flip(img))
for i, img in enumerate(vertical_flip_list):
    ax = plt.subplot(3, 5, i + 1)
    ax.set_title("Flip Vertical")
    plt.imshow(img)
    plt.axis('off')
plt.show()

# Transformation 4 : Ajustement du contraste
contrast_list = []
plt.figure(figsize=(20, 20))
for img in list_img:
    contrast_list.append(adjust_contrast(img))
for i, img in enumerate(contrast_list):
    ax = plt.subplot(3, 5, i + 1)
    ax.set_title("Contraste Adjusté")
    plt.imshow(img)
    plt.axis('off')
plt.show()
#------------------------------------------------------------------------------

SECTION 2 : IMPORT / EXPORT DE DONNÉES
#------------------------------------------------------------------------------
def exercise_import_export():
"""
Regroupe plusieurs exercices d'importation et d'exportation de données.


Collapse
• Exercice 3 : Import du dataset Titanic.
• Exercice 4 : Import du dataset Iris (fichier CSV sans puis avec noms de colonnes).
• Exercice 5 : Export d'un DataFrame vers Excel et JSON.
• Exercice 6 : Lecture de données JSON depuis une URL.
"""
# Exercice 3 : Import du dataset Titanic
print("\n--- Exercice 3 : Import du dataset Titanic ---")
try:
    titanic_df = pd.read_csv('titanic.csv')
    print(titanic_df.head())
except FileNotFoundError:
    print("fichier 'titanic.csv' non trouvé.")

# Exercice 4 : Import du dataset Iris
print("\n--- Exercice 4 : Import du dataset Iris ---")
try:
    iris_df = pd.read_csv('iris.csv', header=None)
    # Assigner des noms de colonnes
    iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    print(iris_df.head())
except FileNotFoundError:
    print("fichier 'iris.csv' non trouvé.")

# Exercice 5 : Export d'un DataFrame vers Excel et JSON
print("\n--- Exercice 5 : Export DataFrame vers Excel et JSON ---")
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [24, 27, 22],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
try:
    df.to_excel('output.xlsx', index=False)
    df.to_json('output.json', orient='records')
    print("DataFrame exporté vers 'output.xlsx' et 'output.json'.")
except Exception as e:
    print("Erreur lors de l'export :", e)

# Exercice 6 : Lecture de données JSON depuis une URL
print("\n--- Exercice 6 : Lecture de données JSON depuis une URL ---")
url = 'https://jsonplaceholder.typicode.com/posts'
try:
    response = requests.get(url)
    json_data = response.json()
    json_df = pd.DataFrame(json_data)
    print(json_df.head())
except Exception as e:
    print("Erreur lors de la lecture du JSON :", e)
#------------------------------------------------------------------------------

SECTION 3 : EXERCICES XP SUR LES DONNÉES STRUCTURÉES VS NON STRUCTURÉES
#------------------------------------------------------------------------------
def exercise_identifying_data_types():
"""
Affiche des exemples identifiant les types de données.
• Rapport sur des données structurées (ex: rapports financiers, inventaire) et non structurées (photos, articles, interviews).
"""
print("\n--- Exercise 1 : Identifying Data Types ---")
examples = {
"Rapports financiers Excel": "Structured data (données organisées en lignes et colonnes).",
"Photographies sur une plateforme sociale": "Unstructured data (images sans modèle prédéfini).",
"Articles de presse": "Unstructured data (texte libre, varié).",
"Données d'inventaire d'une base relationnelle": "Structured data (tables avec des attributs fixes).",
"Interviews enregistrées lors d'une étude de marché": "Unstructured data (audio/vidéo sans format fixe)."
}
for key, value in examples.items():
print(f"{key}: {value}")

def exercise_transformation_text_data():
"""
Exercice 2 : Transformation de données textuelles non structurées en données structurées.
– Par exemple, extraire des informations clés à partir d'une série d'articles de blog ou d'enregistrements audio transcrits.
"""
print("\n--- Exercise 2 : Transformation Textuelle ---")
print("Exemple : Extraire 'Location', 'Activity', 'Date' à partir d'un texte de blog de voyage.")
print("Exemple : Transcrire des appels de service client et extraire les plaintes ou questions pour construire un tableau d'analyse.")

def exercise_retail_unstructured_analysis():
"""
XP Gold – Exercise 1 : Analyse comparative entre données structurées et non structurées.
Charge un dataset de ventes (exemple fictif) et un dataset de reviews.
"""
print("\n--- XP Gold - Exercise 1 : Analyse Comparative ---")
try:
retail_data = pd.read_csv('RetailDataset.csv')
sales_data = retail_data['sales']
print("Analyse des données structurées (ventes) :")
print("Moyenne des ventes :", sales_data.mean())
print("Top 5 ventes :", sales_data.nlargest(5))
except FileNotFoundError:
print("Fichier 'RetailDataset.csv' introuvable.")


try:
    reviews_data = pd.read_csv('WomensClothingReviews.csv')
    reviews = reviews_data['Review Text']
    print("\nAnalyse des données non structurées (reviews) :")
    print("Nombre total de reviews :", len(reviews))
    print("Exemple de reviews :")
    print(reviews.sample(5))
except FileNotFoundError:
    print("Fichier 'WomensClothingReviews.csv' introuvable.")
def exercise_ecommerce_exploration():
"""
XP Gold – Exercise 2 : Exploration de données E-Commerce.
Charge et affiche les premières lignes d'un dataset e-commerce et montre les colonnes structurées.
"""
print("\n--- XP Gold - Exercise 2 : E-Commerce Exploration ---")
try:
ecommerce_data = pd.read_csv('ecommerce_data.csv')
print(ecommerce_data.head())
print("\nColonnes du dataset :", list(ecommerce_data.columns))
except FileNotFoundError:
print("Fichier 'ecommerce_data.csv' introuvable.")

def exercise_public_transport_analysis():
"""
XP Gold – Exercise 3 : Analyse d’un dataset de trafic routier.
"""
print("\n--- XP Gold - Exercise 3 : Analyse de Trafic ---")
try:
traffic_data = pd.read_csv('traffic_volume.csv')
print("Aperçu du dataset trafic :")
print(traffic_data.head())
print("Colonnes identifiées :", list(traffic_data.columns))
except FileNotFoundError:
print("Fichier 'traffic_volume.csv' introuvable.")

def exercise_movie_ratings():
"""
XP Gold – Exercise 4 : Analyse d’un dataset de notation de films.
"""
print("\n--- XP Gold - Exercise 4 : Analyse de Notations de Films ---")
try:
ratings_data = pd.read_csv('ratings.csv')
print("Aperçu du dataset ratings:")
print(ratings_data.head())
print("Colonnes structurées :", list(ratings_data.columns))
except FileNotFoundError:
print("Fichier 'ratings.csv' introuvable.")

def exercise_synthetic_product_catalog():
"""
XP Gold – Exercise 5 : Création d’un catalogue produit synthétique à l'aide de Faker.
"""
print("\n--- XP Gold - Exercise 5 : Catalogue Produit Synthétique ---")
faker = Faker()
products = [{'Product ID': i,
'Name': faker.catch_phrase(),
'Description': faker.text(max_nb_chars=100),
'Price': faker.random_number(digits=5)} for i in range(1, 501)]
product_df = pd.DataFrame(products)
print(product_df.head())
# Optionnel : product_df.to_csv('synthetic_product_catalog.csv', index=False)

def exercise_structured_unstructured_sentiment():
"""
XP Ninja – Exercise 1 : Analyse comparative de données structurées vs non structurées.
Charge un dataset fictif de ventes et un dataset d’emails (exemple) en appliquant une analyse de sentiment.
"""
print("\n--- XP Ninja - Exercise 1 : Analyse Comparative Sentiment ---")
# Partie structured
try:
sales_data = pd.read_csv('product_sales_data.csv')
print("Données structurées (ventes) :")
print(sales_data.describe())
except FileNotFoundError:
print("Fichier 'product_sales_data.csv' introuvable.")
# Partie non structurée
try:
email_data = pd.read_csv('customer_support_emails.csv')
# Extraction d'un sentiment simple avec TextBlob pour chaque email (colonne 'email')
email_data['sentiment'] = email_data['email'].apply(lambda x: TextBlob(x).sentiment.polarity)
print("\nDonnées non structurées (emails) avec sentiment :")
print(email_data.head())
except FileNotFoundError:
print("Fichier 'customer_support_emails.csv' introuvable.")

def exercise_tweets_sentiment_extraction():
"""
XP Ninja – Exercise 2 : Extraction de hashtags et mentions dans un dataset de tweets, puis analyse du sentiment global.
"""
print("\n--- XP Ninja - Exercise 2 : Analyse de Tweets ---")
def extract_hashtags(text):
"""Retourne la liste des hashtags dans le texte."""
return re.findall(r"#(\w+)", text)
def extract_mentions(text):
"""Retourne la liste des mentions dans le texte."""
return re.findall(r"@(\w+)", text)


try:
    tweets_df = pd.read_csv('twitter-tweets-sentiment-dataset.csv')
    # On ajoute deux colonnes pour hashtags et mentions
    tweets_df['hashtags'] = tweets_df['tweet'].apply(extract_hashtags)
    tweets_df['mentions'] = tweets_df['tweet'].apply(extract_mentions)
    # On affiche un résumé
    sentiment_counts = tweets_df['sentiment'].value_counts() if 'sentiment' in tweets_df.columns else "Colonne 'sentiment' absente"
    top_hashtags = tweets_df['hashtags'].explode().value_counts().head(10)
    print("Distribution des sentiments:")
    print(sentiment_counts)
    print("\nTop 10 hashtags:")
    print(top_hashtags)
except FileNotFoundError:
    print("Fichier 'twitter-tweets-sentiment-dataset.csv' introuvable.")
#------------------------------------------------------------------------------

SECTION 4 : SOLUTIONS DIVERSES / KNOWLEDGE CHECKS (AFFICHAGE EXPLICATIF)
#------------------------------------------------------------------------------
def solution_checks():
"""
Affiche quelques réponses théoriques aux questions posées dans les solutions lessons.
"""
print("\n--- Knowledge Check et Explications ---")
print("• Pourquoi Pandas est un outil polyvalent pour l'import de données ?")
print(" ⇒ Il offre des DataFrames et Series robustes, supporte de nombreux formats (CSV, Excel, JSON, SQL, etc.) et permet une manipulation efficace de gros volumes de données.")
print("\n• Avantages de Kaggle :")
print(" ⇒ Large variété de datasets, haute qualité, communauté active et possibilité de compétitions, facilitant l'apprentissage et la collaboration.")
print("\n• Formats de données :")
print(" ⇒ CSV est simple et convient aux données tabulaires, JSON est idéal pour des structures imbriquées, et Excel permet de gérer plusieurs feuilles avec un formatage riche.")
print("\n• Pour importer depuis Google Drive, l'avantage est la persistance, la capacité de stockage étendue et la facilité d'accès via Colab.")

#------------------------------------------------------------------------------

MAIN
#------------------------------------------------------------------------------
def main():
print("=== SECTION 1 : Image Transformation and Visualization ===")
image_transformation_and_visualization()


Collapse
print("\n=== SECTION 2 : Import / Export de Données ===")
exercise_import_export()

print("\n=== SECTION 3 : Exercices XP sur Données Structurées/Non Structurées ===")
exercise_identifying_data_types()
exercise_transformation_text_data()
exercise_retail_unstructured_analysis()
exercise_ecommerce_exploration()
exercise_public_transport_analysis()
exercise_movie_ratings()
exercise_synthetic_product_catalog()
exercise_structured_unstructured_sentiment()
exercise_tweets_sentiment_extraction()

print("\n=== SECTION 4 : Solutions et Knowledge Checks ===")
solution_checks()
if name == "main":
main()