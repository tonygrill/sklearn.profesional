import pandas as pd 

# vamos a comparar estos dos modelos
from sklearn.linear_model import(RANSACRegressor, HuberRegressor)

# tambien lo vamos a comparar con un modelo de basado en maquinas de soporte vectorial
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('felicidad_corrupt.csv')
    print(dataset.tail(5))

    # de nuestros features eliminamos el pais porque es texto  
    #y no aporta nada y el score que queremos predecir
    X = dataset.drop(['country', 'score'], axis=1)

    # nuestro target a predecir será la columna Score
    y = dataset[['score']]

    # procedemos a hacer la partición de datos en entrenamiento y prueba
    # si queremos replicabilidaden nuestro modelo y quee nuestra partición aleatoria sea siempre la misma 
    # debemos usar random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # la manera mas profesional es guardar nuestros estimadores en un diccionario

    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    # las funciones en python pueden retornar mas de un valor
    # vamos a aprovechar esta característica
    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)
        print('='*64)
        print(name)
        print('MSE: ', mean_squared_error(y_test, predictions))