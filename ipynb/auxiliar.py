from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import timeseriesmetrics as tss
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from krlst_implement import Krls_t


def calcular_metricas_regressao(y_true, y_pred):
    # Certifique-se de que y_true e y_pred são arrays de uma única dimensão
    y_true = np.array(y_true).flatten()  
    y_pred = np.array(y_pred).flatten()
    
    # Calcula as métricas
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)  # Raiz do MSE
    mape = tss.mape(y_true, y_pred)
    theil = tss.theil(y_true, y_pred)
    arv = tss.arv(y_true, y_pred)
    wpocid = tss.wpocid(y_true, y_pred)
    
    # Verifica se há mais de um ponto para calcular o R²
    if len(y_true) > 1:
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))  # Coeficiente de determinação
    else:
        r2 = 'N/A'  # Não aplicável para previsões de um único ponto
    
    # Cria um DataFrame com as métricas
    metrics_df = pd.DataFrame({
        'Métrica': ['MAE', 'MSE', 'MAPE','RMSE', 'R^2','THEIL','ARV','WPOCID'],
        'Valor': [mae, mse, mape, rmse, r2, theil, arv, wpocid]
    })
    
    return metrics_df


def getSignificantLags(y, nLags = 5, alpha=0.05):
    pac, pac_ci = sm.tsa.pacf(x=y, nlags=nLags, alpha=alpha)
    pd.DataFrame(pac)[1:].plot(kind='bar', legend='pacf',
                               xlabel='lag', ylabel='pacf');
    significantLags = []
    for i in range(1, len(pac)):
        # print(pac[i], pac_ci[i][0], pac_ci[i][1])
        if pac[i] < pac_ci[i][0] - pac[i] or pac[i] > pac_ci[i][1] - pac[i]:
            significantLags.append(i)
    print('significantLags:', significantLags)
    return significantLags

def appendLagVariables(yNm, significantLags, df, dropna=True):
    prefix = yNm.replace(')', '')
    df = df.copy()
    for lag in significantLags:
        varNm = '('+prefix+'-'+str(lag)+')'
        # nDj = pd.concat([nDj, nDj[['e1(t)']].shift(lag)], axis=1)
        df[varNm] = df[yNm].shift(lag)
        # nDj.columns = nDj.columns + [varNm]
    if dropna:
        df.dropna(axis=0, inplace=True)
    print(df.head(2))
    return df

def sliding_window_cross_validation(X, y, window_size, test_size, params):
    """
    Realiza validação cruzada com janela deslizante para uma árvore de decisão.

    Args:
        X (pd.DataFrame): As variáveis preditoras.
        y (pd.Series): A variável dependente.
        window_size (int): Tamanho da janela de treino.
        test_size (int): Tamanho da janela de teste.

    Returns:
        dict: Um dicionário com as previsões e o erro médio quadrático.
    """
    predictions = []
    actuals = []

    max_depth, min_samples_split, min_samples_leaf, max_features  = params

    # Loop sobre os dados com a janela deslizante
    for start in range(0, len(y) - window_size - test_size + 1):
        # Definir a janela de treino e teste
        X_train = X.iloc[start:start + window_size]
        y_train = y.iloc[start:start + window_size]
        X_test = X.iloc[start + window_size:start + window_size + test_size]
        y_test = y.iloc[start + window_size:start + window_size + test_size]

        # Criar e treinar o modelo de árvore de decisão
        model = DecisionTreeRegressor(
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features
        )
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_test)

        # Armazenar as previsões e os valores reais
        predictions.extend(y_pred)
        actuals.extend(y_test)

    # Calcular o erro médio quadrático
    mse = mean_squared_error(actuals, predictions)

    return {
        'predictions': predictions,
        'actuals': actuals,
        'mse': mse
    }

def rolling_window_cross_validation(X, y, train_size, test_size, params):
    """
    Realiza validação cruzada com rolling window para uma árvore de decisão.

    Args:
        X (pd.DataFrame): As variáveis preditoras.
        y (pd.Series): A variável dependente.
        train_size (int): Tamanho da janela de treino.
        test_size (int): Tamanho da janela de teste.
        params (tuple): Parâmetros para a árvore de decisão (max_depth, min_samples_split, min_samples_leaf, max_features).

    Returns:
        dict: Um dicionário com as previsões e o erro médio quadrático.
    """
    predictions = []
    actuals = []

    max_depth, min_samples_split, min_samples_leaf, max_features = params

    # Loop sobre os dados com a janela rolante
    for start in range(0, len(y) - train_size - test_size + 1):
        # Definir a janela de treino e teste
        X_train = X.iloc[start:start + train_size]
        y_train = y.iloc[start:start + train_size]
        X_test = X.iloc[start + train_size:start + train_size + test_size]
        y_test = y.iloc[start + train_size:start + train_size + test_size]

        # Criar e treinar o modelo de árvore de decisão
        model = DecisionTreeRegressor(
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features
        )
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_test)

        # Armazenar as previsões e os valores reais
        predictions.extend(y_pred)
        actuals.extend(y_test)

    # Calcular o erro médio quadrático
    mse = mean_squared_error(actuals, predictions)

    return {
        'predictions': predictions,
        'actuals': actuals,
        'mse': mse
    }

def blocked_cross_validation(X, y, train_size, test_size, buffer_size, params):
    """
    Realiza validação cruzada com Blocked Cross-Validation para uma árvore de decisão,
    utilizando uma janela de buffer entre os dados de treino e teste.

    Args:
        X (pd.DataFrame): As variáveis preditoras.
        y (pd.Series): A variável dependente.
        train_size (int): Tamanho da janela de treino.
        test_size (int): Tamanho da janela de teste.
        buffer_size (int): Tamanho da janela de buffer entre treino e teste.
        params (tuple): Parâmetros para a árvore de decisão (max_depth, min_samples_split, min_samples_leaf, max_features).

    Returns:
        dict: Um dicionário com as previsões e o erro médio quadrático.
    """
    predictions = []
    actuals = []

    max_depth, min_samples_split, min_samples_leaf, max_features = params

    # Loop sobre os dados com a janela bloqueada
    for start in range(0, len(y) - train_size - buffer_size - test_size + 1):
        # Definir a janela de treino
        X_train = X.iloc[start:start + train_size]
        y_train = y.iloc[start:start + train_size]

        # Definir a janela de teste após o buffer
        X_test = X.iloc[start + train_size + buffer_size:start + train_size + buffer_size + test_size]
        y_test = y.iloc[start + train_size + buffer_size:start + train_size + buffer_size + test_size]

        # Criar e treinar o modelo de árvore de decisão
        model = DecisionTreeRegressor(
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features
        )
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_test)

        # Armazenar as previsões e os valores reais
        predictions.extend(y_pred)
        actuals.extend(y_test)

    # Calcular o erro médio quadrático
    mse = mean_squared_error(actuals, predictions)

    return {
        'predictions': predictions,
        'actuals': actuals,
        'mse': mse
    }

## ADPTANDO PARA O KRLS-T

# lambda_= 0.0001, c=0.00001,M=30, sigma = 1
def sliding_window_krlst_cross_validation(X, y, window_size, test_size, params):
    """
    Realiza validação cruzada com janela deslizante para uma árvore de decisão.

    Args:
        X (pd.DataFrame): As variáveis preditoras.
        y (pd.Series): A variável dependente.
        window_size (int): Tamanho da janela de treino.
        test_size (int): Tamanho da janela de teste.

    Returns:
        dict: Um dicionário com as previsões e o erro médio quadrático.
    """
    predictions = []
    actuals = []

    lambda_, c, M, sigma  = params

    # Loop sobre os dados com a janela deslizante
    for start in range(0, len(y) - window_size - test_size + 1):
        # Definir a janela de treino e teste
        X_train = X.iloc[start:start + window_size]
        y_train = y.iloc[start:start + window_size]
        X_test = X.iloc[start + window_size:start + window_size + test_size]
        y_test = y.iloc[start + window_size:start + window_size + test_size]

        # Criar e treinar o modelo de árvore de decisão
        model = Krls_t(lambda_, c, M, sigma)
        # partial fit
        for i in range(len(X_train)):
            model.learn_one(X_train.iloc[i], y_train.iloc[i], (X_train.index[i]))

        # Fazer previsões
        y_pred = []
        for j in range(len(X_test)):
            y_pred_aux, desv = model.predict(X_test.iloc[j])
            model.learn_one(X_test.iloc[j], y_test.iloc[j], int(X_test.index[j]))
            #y_pred_aux = y_pred_aux.flatten() # APARENTEMENTE CONTÉM UM BUG AQUI --> VERIFICAR <---
            y_pred.append(y_pred_aux)
        y_pred = [arr[0, 0] for arr in y_pred]
        y_pred = (np.array(y_pred)).reshape(-1, 1)


        # Armazenar as previsões e os valores reais
        predictions.extend(y_pred)
        actuals.extend(y_test)


    # Calcular o erro médio quadrático
    mse = mean_squared_error(actuals, predictions)

    return {
        'predictions': predictions,
        'actuals': actuals,
        'mse': mse
    }