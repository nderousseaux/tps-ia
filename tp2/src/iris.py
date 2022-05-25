import pandas as pd
from sklearn.model_selection import train_test_split
from NeuralNet import NeuralNet
from utility import Utility

iris_df = pd.read_csv("./iris.csv")
df_columns = iris_df.columns.values.tolist()

#On sépare les attributs des étiquettes
features = df_columns[0:4] # 4 premi`eres colonnes
label = df_columns[4:]
X = iris_df[features]
y = iris_df[label]

#On encode les labels
y = pd.get_dummies(y)

#On sépare le jeu d'entrainement et le jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

n = NeuralNet(
    X_train,
    y_train,
    X_test,
    y_test,
    [3,2],
    Utility().identity
)
n.one_epoch()
pass