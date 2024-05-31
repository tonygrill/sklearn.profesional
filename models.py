import pandas as pd
import numpy as np 

# Del modulo support vector machine importamos support  vector regression
from sklearn.svm import SVR

# del modulo de ensamblado importamos un método de ensamble  
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

# al final vamoa a necesitar exportar nustro modelo
from utils import Utils

# creamos la clase

class Models:
    
    def __init__(self):   # definimos un dicc de los modelos que vamos a utilizar
        self.reg = {
            'SVR' : SVR(),
            'GRADIENT' : GradientBoostingRegressor()
        }
            #declaramos nuestro dicc de parametros para nuestro modelo de soporte vectorial
        self.params = {
            'SVR' : {
                'kernel' : ['linear', 'poly', 'rbf'],
                'gamma' : ['auto', 'scale'],
                'C' : [1,5,10]
            }, 'GRADIENT' : {
                'loss' : ['square_error', 'absolute_error'],
                'learning_rate' : [0.01, 0.05, 0.1]
            }
        }
    
    def grid_training(self, X, y):

        best_score = 999
        best_model = None

        for name, reg in self.reg.items():
            # a nuestro gris le pasamos el dicc reg y el dicc de parametros
            grid_reg = GridSearchCV(reg, self.params[name], cv= 3).fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)

            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_

        utils = Utils()
        utils.model_export(best_model, best_score)