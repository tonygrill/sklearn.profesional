import pandas as pd 
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#para dividir los datos en entrenamiento y prueba 
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./felicidad.csv')
    print(dataset.describe())
    print(dataset.info())

    # procedemos a dividor los datos en features 'X' y target 'y'
    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    print(X.shape)
    print(y.shape)

    #ahora dividiremos en train y test
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

    model_linear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = model_linear.predict(X_test)

    model_Lazzo = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_Lazzo = model_Lazzo.predict(X_test)

    model_Ridge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_Ridge = model_Ridge.predict(X_test)


    #tenemos en cuenta de que mientras  menor se la perdida mejor sera el modelo
    #ahora calculamos nuestra perdida lineal
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print('Linear Loss', linear_loss)

    #calculamos la perdida Lasso
    lasso_loss = mean_squared_error(y_test, y_predict_Lazzo)
    print('Lasso loss', lasso_loss)

    #calculamos la perdida ridge
    ridge_loss = mean_squared_error(y_test, y_predict_Ridge)
    print('Ridge Loss', ridge_loss)

    print("="*32)
    print('Coef LASSO')
    print(model_Lazzo.coef_)

    print("="*32)
    print('Coef Ridge')
    print(model_Ridge.coef_)
