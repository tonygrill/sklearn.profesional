import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')

    #print(dt_heart.head(10))

    # eliminamos la columna target de features y creamos una nueva variable solo con la columna target
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    # aqui estandarizamos todas las columnas con valores de 0 y 1
    dt_features = StandardScaler().fit_transform(dt_features)
    
    # aqui partimos el conjunto de entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    # verificamos que el modelo tenga el mismo tamaño
    print(X_train.shape)
    print(y_train.shape)

    # aqui vamos a reducir el número de de columnas a 3, las mas determinantes
    pca = PCA(n_components=3)
    pca.fit(X_train)

    # aqui hacemos lo mismo pero IPCA lo va a dividir en 10 partes para con colapsar la memoria
    ipca = IncrementalPCA(n_components=3, batch_size=10)

    # imprimimmos
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    # creamos una instancia del clasificador de regresión logística.
    logistic =  LogisticRegression(solver='lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print('SCORE PCA: ', logistic.score(dt_test, y_test))

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print('SCORE IPCA: ',logistic.score( dt_test, y_test))