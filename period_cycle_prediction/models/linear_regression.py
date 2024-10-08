import sys
sys.path.append("../period_cycle_prediction/")

from period_cycle_prediction.utils import utils         
import numpy as np #Para trabalhar com arrays
import pandas as pd
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.linear_model import LinearRegression #Para importar o modelo de regressão linear
from sklearn.model_selection import train_test_split #Para dividir o dataset em treino e teste
from sklearn.metrics import mean_squared_error
from math import sqrt
import pendulum
from pendulum import duration

if __name__ == '__main__':
    # Abrir dataset sintético
    df =  pd.read_csv('period_cycle_prediction/dataset/synthetic_data.csv', sep=',', header=0)
    data_years = utils.calculate_cycle_and_periods(df)
    # Preparando os dados para o modelo de regressão linear/machine learning
    periods_data = utils.calculate_datatime(df)
    features, labels = utils.generate_final_features(df)



    x_train, x_test, y_train, y_test  = train_test_split(features, labels, test_size=0.2, random_state=10) 
    # Redefinindo os dados para o modelo de regressão linear/machine learning
    train_y=np.array(y_train)
    train_x= np.array(x_train)
    test_x= np.array(x_test)
    test_y= np.array(y_test)
    train_x = train_x.reshape((train_x.shape[0],train_x.shape[1]*train_x.shape[2]))
    train_y = train_y.reshape((train_y.shape[0],train_y.shape[1]*1))
    test_x = test_x.reshape((test_x.shape[0],test_x.shape[1]*test_x.shape[2]))
    test_y = test_y.reshape((test_y.shape[0],test_y.shape[1]*1))
    train_size = train_x.shape[0]
    # Modelo de regressão linear
    model_LR= LinearRegression()
    model_LR.fit(train_x, train_y)
    # Fazer a predição
    y_pred = model_LR.predict(test_x)
    output_pred = [[int(round(i[0])), int(round(i[1]))] for i in y_pred] # round the values 

    cycle_length=[]
    periods=[]
    for i in range(len(output_pred)):
        cycle_length.append(output_pred[i][0] )
        periods.append(output_pred[i][1] )

    # predição um passo a frente / novo ciclo
    predicao_um_passo_a_frente = model_LR.predict([test_x[-1]])
    cycles_numbers = np.arange(1, len(cycle_length)+1)

    last_know_data_cycle = (data_years)[train_x.shape[0]]

    next_period = utils.data_formatting_prediction(last_know_data_cycle, predicao_um_passo_a_frente)

    width = 8
    height = 5
    plt.figure(figsize=(width, height))
    plt.rcParams.update({'font.size': 16})

    plt.plot(cycle_length, '-->', color='blue', linewidth=3.0)
    plt.plot(test_y[:,0], ':o', color='red', linewidth=3.0 )
    plt.plot(cycles_numbers[-1], predicao_um_passo_a_frente[0][0], '-->', color='blue')
    plt.legend(['Predito',  'Real', 'Predição um passo a frente'])
    plt.grid()
    plt.xlabel('Ciclos')
    plt.ylabel('Dias')
    plt.title('Variação dos Ciclos') 
    fig = plt.gcf()
    fig.savefig('linear.png', dpi=300, bbox_inches='tight')
    plt.show()  

    plt.figure(figsize=(width, height))
    plt.plot(periods, '-*', color='green', linewidth=3.0)
    plt.plot(test_y[:,1], ':o', color='red', linewidth=3.0 )
    plt.plot(cycles_numbers[-1], predicao_um_passo_a_frente[0][1], '-*', color='green')
    plt.legend(['Predito', 'Real', 'Predição um passo a frente'])
    plt.grid()
    plt.xlabel('Períodos')
    plt.ylabel('Dias')
    plt.title('Variação dos Períodos')
    fig = plt.gcf()
    fig.savefig('linear_periods.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(width, height))
    output_pred = [[int(round(i[0])), int(round(i[1]))] for i in y_pred] # round the values
    error = abs(test_y-output_pred)
    plt.plot(error[:,0], '-->', color='blue')
    plt.legend(['Erro Ciclo'])
    plt.grid()
    plt.xlabel('Ciclos')
    plt.ylabel('Dias')
    plt.title('Modelo Regressão Linear') 
    fig = plt.gcf()
    fig.savefig('linear_error.png', dpi=300, bbox_inches='tight')
    plt.show() 

    plt.figure(figsize=(width, height))

    plt.plot(error[:,1], '-*', color='green')
    plt.legend(['Erro Período'])
    plt.grid()
    plt.xlabel('Períodos')
    plt.ylabel('Dias')
    plt.title('Modelo Regressão Linear')
    fig = plt.gcf()
    fig.savefig('linear_error_periods.png', dpi=300, bbox_inches='tight')
    plt.show()


    
    # calcular o RMSE 
    rms = sqrt(mean_squared_error(test_y, y_pred))
    print('RMSE: ', rms)

    # calcular o MAE (Mean Absolute Error)
    mae = np.mean(np.abs((test_y - y_pred)))
    print('MAE: ', mae)
