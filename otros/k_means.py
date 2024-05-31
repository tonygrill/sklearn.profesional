# CLUSTERING    

import pandas as pd
from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':

    dataset = pd.read_csv('candy.csv')
    print(dataset.head(5))

    # como no estamos usando aprendizaje supervizado no es necesario dividir los datos 
    # en entrenamiento y prueba

    X = dataset.drop('competitorname', axis=1)

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print('Total de centros: ', len(kmeans.cluster_centers_))
    print('=', 64)
    print(kmeans.predict(X))

    # creamos una nueva columna con la predicción
    dataset['group'] = kmeans.predict(X)

    print(dataset.tail(10))