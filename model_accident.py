from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ===  Lecture des fichiers ===

def lire_fichier_csv(chemin_fichier, skip_header=True):
    data_csv =  []
    with open(chemin_fichier) as fic:
        lines = fic.readlines()
        data_csv = [line.strip().split(";") for line in lines]
        if skip_header:
            data_csv = data_csv[1:]
        return data_csv

data_usagers   =  lire_fichier_csv("data/usagers-2023.csv")
data_vehicules =  lire_fichier_csv("data/vehicules-2023.csv")

def convert_grav(val) :
    """
    1 – Indemne -> 1
    2 – Tué -> 100
    3 – Blessé hospitalisé -> 10
    4 – Blessé léger -> 5
    autre -> -1
    """
    if val == "1":
        return 1
    elif val == "2":
        return 100
    elif val == "3":
        return 10
    elif val == "4":
        return 5
    return -1

def convert_annee(val):
    if len(val) != 4:
        return -1
    return int(val, 10)

# Suppression des lignes qui sont trop courtes
data_usagers = [ d for d in data_usagers if len (d) > 8]

# Conversion des données
xy = [[
                convert_annee(d[8][1:-1]),
                convert_grav(d[6][1:-1])
            ]
            for d in data_usagers]

# Elimination des données incohérentes
xy = [d for d in xy if d[0] > -1 and d[1] > -1]

x_annee = [xy[0] for xy in xy]
y_gravite = [xy[1] for xy in xy]

print("Taille des données", "en entrée :", len(data_usagers), "retenus :", len(x_annee), len(y_gravite))

