from os import stat

from ex1 import count_occurences, entropy

class Node:

    def __init__(self, occ=0, car=None, leftNode=None, rightNode=None):
        self.occ = occ
        self.car = car
        self.leftNode = leftNode
        self.rightNode = rightNode

    def isFeuille(self):
        if self.leftNode == None and self.rightNode == None:
            return True
        return False

def optimum_tree(occurences):
    liste_noeuds = [Node(occ, car) for (car, occ) in occurences.items()] 

    liste_noeuds.sort(key=lambda node:node.occ)

    while len(liste_noeuds) > 1:
        n1 = liste_noeuds.pop(0)
        n2 = liste_noeuds.pop(0)
        liste_noeuds.append(Node(n1.occ + n2.occ, None, n1, n2))
        liste_noeuds.sort(key=lambda node:node.occ)

    return liste_noeuds[0]

# renvoie la manière de coder un caractère
def code(arbre, car):
    if arbre.car == car:
        return ""

    left = -1
    right = -1

    if arbre.leftNode != None:
        left = code(arbre.leftNode, car)
    
    if arbre.rightNode != None: 
        right = code(arbre.rightNode, car)

    if left == -1 and right == -1:
        return -1

    if right != -1:
        return "1"+right

    if left != -1:
        return "0"+left
    

# Calcule le nombre de bit moyens pour coder le texte
def bit_moyens(arbre, occurences):
    return sum([len(code(arbre, car)) for car in occurences])/len(occurences)

def compression(file_path, arbre):
    occurences = count_occurences(file_path)

    symboles = dict([(car, code(arbre, car)) for car in occurences])

    texte = ""

    with open(file_path, 'r') as f:
        for ligne in f:
            for car in ligne:
                texte+=symboles[car]

    return texte

def decompresssion(texte, arbre):

    texte_decompresse = ""

    noeud_courrant = arbre
    for car in texte:
        if car == '0':
            noeud_courrant = noeud_courrant.leftNode
        else:
            noeud_courrant = noeud_courrant.rightNode

        if noeud_courrant.isFeuille():
            texte_decompresse+=noeud_courrant.car
            noeud_courrant = arbre

    return texte_decompresse

def question3_4():
    # tonton
    occurences = count_occurences("./tonton.txt")
    arbre_tonton = optimum_tree(occurences)
    en = entropy(occurences)
    bits = bit_moyens(arbre_tonton, occurences)
    print("Tonton : nombre de bits moyen : {}, entropie : {}".format(bits, en))

    # Chasseur
    occurences = count_occurences("./chasseur.txt")
    arbre_chasseur = optimum_tree(occurences)
    en = entropy(occurences)
    bits = bit_moyens(arbre_chasseur, occurences)
    print("Chasseur : nombre de bits moyen : {}, entropie : {}".format(bits, en))

    #On compresse/décompresse le texte tonton
    texte = compression('./tonton.txt', arbre_tonton)
    print(texte)
    texte_decompresse = decompresssion(texte, arbre_tonton)
    print(texte_decompresse)

    #On compresse/décompresse le texte chasseur
    texte = compression('./chasseur.txt', arbre_chasseur)
    print(texte)
    texte_decompresse = decompresssion(texte, arbre_chasseur)
    print(texte_decompresse)

path = './oeuvres_rousseau.txt'
occurences = count_occurences(path)
arbre = optimum_tree(occurences)
en = entropy(occurences)
texte_compresse = compression(path, arbre)
taille = stat(path).st_size * 8 
taux_compression = len(texte_compresse)/taille
print("Le texte compressé prend {} bits, le texte décompréssé prend {}. Donc le taux de compression est de : {}%"
    .format(
        len(texte_compresse),
        taille,
        round(taux_compression * 100)
    )
)