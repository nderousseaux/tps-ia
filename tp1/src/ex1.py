from math import log2

def count_occurences(file_path):
    occurences = {}

    f = open(file_path, 'r')
    for ligne in f:
        for car in ligne:
            if not car in occurences:
                occurences[car] = 0
            occurences[car]+= 1
    f.close()

    return occurences

def entropy(occurences):
    res = 0

    total = sum(occurences.values())
    for car in occurences:
        p = occurences[car]/total
        res += p*log2(p)


    return -res