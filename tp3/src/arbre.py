import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import numpy as np
# sns.pairplot(iris_dataframe , hue="class")
# plt.show() 

def construction_arbre(data, cible, attributs_restant, profondeur):
    attribtut, gain, split, partitions = meilleur_attribut(data, attributs_restant[:-1]) #On split les données
    prediction = data[cible].value_counts()
    if profondeur > 4 or gain == 0:
        return node(attribtut, split, prediction)
    
    right = construction_arbre(partitions[1], cible, attributs_restant, profondeur+1)
    left = construction_arbre(partitions[0], cible, attributs_restant, profondeur+1)

    return node(attribtut, split, prediction, right, left)


class node():
     
    # def __init__(self, a, split, predic, right=None, left=None):
    #     self.attribute = a
    #     self.split = split
    #     self.predic = predic
    #     self.right = right
    #     self.left = left 

    def node_result(self, spacing):
        se = 0
        ve = 0
        vergi = 0
        if "Iris-setosa" in self.predic:
            se = self.predic["Iris-setosa"]/sum(self.predic.values)
        if "Iris-versicolor" in self.predic:
            ve = self.predic["Iris-versicolor"]/sum(self.predic.values)
        if "Iris-virginica" in self.predic:
            vergi = self.predic["Iris-virginica"]/sum(self.predic.values)
        print("Setosa : {}, Vernicolor {}, Virginica {}".format(se, ve, vergi))


    # def isLeaf(self):
    #     if self.right == None and self.left == None:
    #         return True
    #     return False

def gain_a(data, partitions):
    res = entropy(data)
    for p in partitions:
        res -= (len(p)/len(data)) * entropy(p)
    return res

def gain_attribute(data, attribute):
    best_gain = 0
    partitions = [0,0]
    #Pour chaque quantile, on calcule le gain d'information
    for quantile in np.arange(0.1, 1, 0.1):
        split_value = data.quantile(quantile)[attribute]
        p1 = data.loc[data[attribute] < split_value]
        p2 = data.loc[data[attribute] >= split_value]
        g = gain_a(data, [p1,p2])
        if g > best_gain:
            best_gain = g
            partitions[0] = p1
            partitions[1] = p2

    return g, split_value, partitions

def meilleur_attribut(df, attribtuts):
    gain = 0
    split_value = 0
    partitions = 0
    attribute = 0
    for attribute_r in attribtuts:
        gain_r, split_value_r, partitions_r = gain_attribute(df, attribute_r)
        if gain_r > gain:
            gain = gain_r
            split_value = split_value_r
            attribute = attribute_r
            partitions = partitions_r

    
    
    return attribute, gain, split_value, partitions

def entropy(df):
    res = 0
    total = df.shape[0]  

    for nb in df["class"].value_counts():
        proba = nb/total
        res += proba * math.log(proba, 2)
    
    return -res

def print_tree(n, spacing=''):
    if n is None:
        return
    if n.isLeaf():
        n.node_result(spacing)
        return
    print("{}[Attribute : {} Split value : {}]".format(spacing, n.attribute, n.split))


    print(spacing + "> True")
    print_tree(n.left, spacing + "-")

    print(spacing + "> False")
    print_tree(n.right, spacing + "-")

def test(arbre, data):

    if arbre.isLeaf(): 
        arbre.node_result("")
        
        return arbre.predic.sort_values(ascending=False).index[0] == data["class"].iloc[0]

    else:
        split_value = arbre.split
        if data[arbre.attribute].iloc[0] < split_value:
            return test(arbre.left, data)
        else:
            return test(arbre.right, data)

iris_dataframe = pd.read_csv("iris.csv")
arbre =construction_arbre(iris_dataframe, "class", iris_dataframe.columns.values, 0)
# print_tree(arbre)

j = 0
z = 150
for i in range(1, 150):
    print("----------------------------------")
    instance = iris_dataframe.sample()
    print(instance["class"].iloc[0])

    a = test(arbre, instance)
    if a:
        j += 1
    
print(j/z)
pass

# en = entropy(iris_dataframe)
# print(data_sorted.quantile(0.25))
# print(data_sorted.quantile(0.5))
# print(data_sorted.quantile(0.75))
# print(data_sorted.quantile(1))


# # Nombre d’instances dans le dataframe:
# nb_lignes = iris_dataframe.shape[0]
# # Decompte du nombre d’instance de chaque classe dans le dataframe
# series =  
# # Sur le dataframe global (l’ensemble des donnees), series retourne:
# # Iris-setosa 50 # Iris-versicolor 50
# # Iris-virginica 50
# # Name: class , dtype: int64
# # Recuperer une valeur associee a une etiquette donnee
# occ_setosa = series.get("Iris-setosa")
# print(occ_setosa)