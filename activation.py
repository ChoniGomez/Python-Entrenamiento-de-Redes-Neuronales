import numpy as np

# Esta es la funcion de activacion de la neurona, en este caso sigmoidal
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Esta es la funcion escalon
def unit_step(x):
    return 1 if x > 0 else 0
    #return (x>0, 1, 0) #Retorna 1 si x >0, sino retorna 0    