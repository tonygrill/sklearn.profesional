import pandas as pd

from sklearn.model_selection import RandomizedSearchCV

# haremos una regresion utilizando un metodo de ensamble
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':

    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.head(5))

    X = dataset.drop(['country', 'rank', 'score'], axis=1)
    y = dataset['score']

    #luego de esto lo primero que debemos definir es el regresor
    #sin parametros en este caso 
    reg = RandomForestRegressor()

    #pasamos los parametros en una grilla
    parametros = {
        'n_estimators' : range(4,16), #es cuantos arboles van a componer mi bosque aleatorio
        'criterion' : ['squared_error', 'absolute_error'], #es una medida de calidad de los split que hace  mi arbol, dice que tan bueno o malo fue
        'max_depth' : range(2,10)  #nos permite establecer que tan profundo es nuestro arbol
    }  # cuando  trabajamos con random forest regressor trabajamos con clasificadores y regresores que son debiles
    #entonces nuestros arboles no van a tener una profundidad muy grande

    
    # n_iter limita el maximo de muestreos que podemos hacer en los parametros
    # cv cross validation toma todo nuestro conjunto de datos y lo divide en 3 pliegues y va a utilizar 2 de training y uno de test
    # 
    rand_est = RandomizedSearchCV(reg, parametros, n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X,y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print('valor predicho: ', rand_est.predict(X.loc[[10]]), 'valor real: ', y.iloc[10])