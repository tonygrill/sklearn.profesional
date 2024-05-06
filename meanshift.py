# CLUSTERING

import pandas as pd
from sklearn.cluster import MeanShift

if __name__ == '__main__':

    dataset = pd.read_csv('candy.csv')
    print(dataset.head(8))

    # eliminamos los nombres ya que es unna columna categórica
    X = dataset.drop(['competitorname'], axis=1)

    # definimos una variable para guardar nuestro modelo
    meanshift = MeanShift().fit(X)

    #imprimimos el numero de etiquetas que el modelo agrupó
    print(max(meanshift.labels_))

    print('='*64) 

    # 
    print(meanshift.cluster_centers_) 

    dataset['meanshift'] = meanshift.labels_
    print('='*64)
    print(dataset)