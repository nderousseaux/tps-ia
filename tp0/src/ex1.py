import numpy as np

#Manipulation de vecteurs / matrices

#1. Créer un vecteur v de 10  éléments entiers aléatoires de 0 à 10.
v = np.random.randint(11, size=10)
# (a) Sélectionner tous les éléments de ce vecteur jusqu’aux 4ème (inclus). 
Q1a = v[:4]
# (b) Sélectionner tous les éléments de ce vecteur après le 4ème.
Q1b = v[4:]
# (c) Sélectionner le dernier élément de ce vecteur.
Q1c = v[-1]
# (d) Sélectionner tous les éléments de ce vecteur sauf le dernier.
Q1d = v[:-1]

#2. Créer une matrice a de 4x3 éléments entiers aléatoires de 0 à 10 inclus.
a = np.random.randint(11, size=[4, 3])
# (a) Sélectionner les 2 premières ligne de la matrice.
Q2a = a[:2]
# (b) Sélectionner les 2 premières colonnes de la matrice.
Q2b = a[:, :2]
# (c) Sélectionner les 2 premières lignes de la 3ème colonne de la matrice. 
Q2c = a[:2, 2]
# (d) Sélectionner le dernier élément de la matrice.
Q2d = a[-1, -1]
# (e) Afficher la forme de la matrice.
Q2e = np.shape(a)
# (f) Afficher le nombre de lignes de la matrice.
Q2f = len(a)
# (g) Afficher le nombre de colonnes de la matrice.
Q2g = len(a[0])
# (h) Afficher le nombre total d’éléments dans la matrice.
Q2h = Q2f*Q2g
Q2hbis = a.size
# (i) Afficher la dimension de la matrice.
Q2i = len(Q2e)
Q2ibis = a.ndim

#3. Manipulation++ de vecteurs et matrices
# (a) Transposer le vecteur précédemment créé.
Q3a = v.T
# (b) Applatir la matrice précédemment créée. Comment est arrangé le vecteur issu de la transformation ?
Q3b = a.flatten() #RES: On a mit bout à bout les lignes
# (c) Redimensionner v de facon à l’afficher comme une matrice de 2 lignes et 5 colonnes.
Q3c = np.resize(v, [2,5])
# (d) Créer une seconde matrice m de mêmes dimensions que la première et contenant des valeurs aléatoires distribuées normalement autour de 2 et avec un écart type de 1. Vous aurez auparavent fixé la graine du générateur de nombre pseudo-aléatoire de numpy.
np.random.seed(2142312)
m = np.random.normal(2, 1.5, size=[4, 3])
# (e) Est-ce possible de multiplier les matrices a et m ? Pourquoi ?
#RES: Non, car la hauteur de l'une ne correspond pas à la longueur de l'autre et inversement
# (f) Créer la matrice m2 qui sera la transposée de m.
m2 = m.transpose()
# (g) Multiplier les matrices a et m2.
Q3g = np.dot(a,m2)
# (h) Calculer la moyenne et l’écart type de m2.
Q3h = (m2.mean(), m2.std())
# (i) Créer deux matrices carrées a1 et a2 de dimensions 3x3 et contenant des entiers aléatoires entre 0 et 10 non inclus.
a1 = np.random.randint(10, size=[3, 3])
a2 = np.random.randint(10, size=[3, 3])
# (j) Sommer a1 et a2.
Q3j = a1+a2
# (k) Soustraire a1 et a2.
Q3k = a1-a2

pass