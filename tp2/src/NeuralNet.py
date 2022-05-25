import numpy as np
from utility import Utility
class NeuralNet:
    W = []
    B = []
    A = []
    df = []
    Z = []

    def __init__(self, X_train = None, y_train = None,
            X_test = None, y_test = None,
            hidden_layer_sizes=(4,),
            activation="identity",
            learning_rate=0.1,
            epoch=200
        ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.layer_sizes =  [len ( X_train.columns )] + hidden_layer_sizes + [len( y_train.columns )] 
        self.initialise_matrices()

    def initialise_matrices(self):
        self.A = [None] * (len(self.hidden_layer_sizes) + 1)
        self.df = [None] * (len(self.hidden_layer_sizes)+1)
        
        for i in range(1, len(self.layer_sizes)):
            #W représente le poids de la connections avec un neurone de la couche précédente
            self.W.append(
                np.random.uniform(
                    low=-1.0, high=1.0,
                    size=(
                        self.layer_sizes[i],
                        self.layer_sizes[i-1]
                    )
                )
            )
            #B représente le biais de chaque neurone donc 3 matrices aussi
            self.B.append(
                    np.zeros(self.layer_sizes[i])
            )

            self.Z = [None] * len(self.layer_sizes)
            self.df = [None]* len(self.layer_sizes)


    def one_epoch(self):
        erreur_train = []
        for idx, Xi in enumerate(self.X_train.iloc):
            y = self.y_train.values[idx]

            erreur = self.prop_forward(Xi.values, y)
            erreur_train.append(erreur)
            self.prop_backward(Xi.values, y)

        erreur_train = erreur_train.sum()/len(erreur_train)
            
    def prop_forward(self, data, y):
        #Z[l] = W[l]A[l−1] +b[l] et A[l] = g(Z[l])

        #Pour chaque couche
        next_layer = data[...,None]; 
        for idx in range(len(self.hidden_layer_sizes)+1):
            self.Z[idx+1] = np.dot(self.W[idx], next_layer) + self.B[idx]
            
            #Si c'est la dernière couche la fonction d'activation est soft max
            if idx != len(self.hidden_layer_sizes)-1:
                self.A[idx]  = self.activation(self.Z[idx+1])[0]
                next_layer = self.A[idx]
            else:
                self.A[idx] = Utility().softmax(self.Z[idx+1])
            
        return Utility().cross_entropy_cost(self.A[idx][...,None],  y[...,None])
        
    def prop_backward(self, X, y):
        delta = [None] * (len(self.hidden_layer_sizes)+1)
        dW    = [None] * (len(self.hidden_layer_sizes)+1)
        db    = [None] * (len(self.hidden_layer_sizes)+1)

        # # #On initilise la dernière couche
        delta[-1] = self.A[-1]-y # valable que pour softmax & entropie crois ́ee! 
        dW[-1] = np.transpose(delta[-1] * self.A[-2])
        db[-1] = delta[-1]

        for l in  range(len(self.hidden_layer_sizes) -1 , -1, -1):
            delta[l] = np.multiply(np.dot(self.W[l+1].T, delta[l+1]), self.df[l])

            if l == 0:
                dW[l] = np.transpose(delta[l] * X[...,None].T)
            else:
                dW[l] = np.transpose(delta[l] * self.A[l-1][...,None].T)
            
            db[l] = delta[l]
        
        for l in range(self.n_hidden_layers + 1):
            self.W[l] = self.W[l] - self.learning_rate*dW[l]
            self.B[l] = self.B[l] - self.learning_rate*db[l]

        pass
        




        

def to_column(x):
  x = np.array(  [[i] for i in x]  )
  return x
