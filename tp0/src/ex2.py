import pandas as pa
import numpy as np

# Manipulation d’un Dataframe

# 1. Importer les données dans un dataframe.
Q1 = pa.read_csv('./src/abalone.csv', sep=',')
# 2. Afficher les 10 premières instances du jeu de données.
Q2 = Q1[:10]
# 3. Quels sont les noms des attributs du jeu de données ?
Q3 = Q1.columns
# 4. Combien d’instance ce jeu de données comporte-t-il ? Manque-t-il des données ?
Q4 = len(Q1)
Q4bis = Q1.isnull().sum().sum()
# 5. Quels sont les types des attributs ?
Q5 = Q1.dtypes
# 6. Considérons que ce que nous souhaitons prédire le nombre d’anneaux des individus selon leurs différentes caractéristiques.
# (a) Combien y’a-t-il de valeurs d’anneaux différentes ?
Q6a = Q1["Rings"].nunique()
# (b) Quels sont les 3 valeurs les plus représentées en nombre d’instances ?
Q6b = Q1["Rings"].value_counts()[:3]
# 7. Considérons maintenant que nous souhaitons prédire le sexe d’un individus selon ses caractéristiques.
# (a) Quelles sont les valeurs possibles de l’attribut correspondant ? 
Q7a = Q1["Sex"].unique()
# (b) Combien d’instances existe-t-il pour chaque valeur possible ?
Q7b = Q1["Sex"].value_counts()
# (c) La répartition des instances parmis ces valeurs possibles vous semble-t-elle équilibrée ?
#RES: Oui
# (d) Transformer les valeurs possibles de l’attribut en un code entier.
Q1["Sex"].replace(['M','F', 'I'],[0,1,2], inplace=True)
# (e) Séparer les données en deux ensembles X (les attributs) et y (les cibles)
y = Q1["Sex"].squeeze()
X = Q1.drop(columns="Sex")
# (f) Transformer votre dataframe X et votre série y en tableaux numpy.
yNumpi = y.to_numpy()
xNumpi = X.to_numpy()
pass