import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import(cross_val_score, KFold)

if __name__ == '__main__':
    dataset = pd.read_csv('felicidad.csv')

    # vamos a usar todas las columnas menos el nombre del país y el score 
    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']

    # creamos nuestro modelo
    model = DecisionTreeRegressor()
    #antes usabamos fit, pero en este caso usaremos cross validation, la cual
    # en la recomendable cuando queremos hacer una prueba rápida
    score = cross_val_score(model, X,y, scoring='neg_mean_squared_error')
    print(score)
    # el  uso de abs (valor absoluto) tiene el propósito de convertir las puntuaciones de error cuadrático medio (MSE) negativas en positivas. 
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(dataset):
        print(train)
        print(test)