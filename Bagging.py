import pandas as pd

#importamos nuestro clasificador
from sklearn.neighbors import KNeighborsClassifier

# importamos el metodo de ensamble
from sklearn.ensemble import BaggingClassifier

from  sklearn.model_selection import train_test_split

# debido a que es una clasificación, necesitamos edr nuestro exito
from sklearn.metrics import accuracy_score 

if __name__ == '__main__':
    dt_heart = pd.read_csv('heart.csv')
    print(dt_heart['target'].describe())

    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    #dividimos los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print('='*64)
    print('KNEIGHBORS: ', accuracy_score(knn_pred, y_test))

    bagg_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bagg_predd = bagg_class.predict(X_test)
    print('='*64)
    print('BAGGING: ', accuracy_score(bagg_predd, y_test))