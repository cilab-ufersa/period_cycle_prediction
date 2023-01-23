import sys
sys.path.append("../period_cycle_prediction/")

from period_cycle_prediction.utils import utils         
import numpy as np #Para trabalhar com arrays
import pandas as pd
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.ensemble import RandomForestRegressor #Para importar o modelo 
from sklearn.model_selection import train_test_split #Para dividir o dataset em treino e teste
from sklearn.metrics import mean_squared_error
from math import sqrt

if __name__ == '__main__':
    # Abrir dataset sintético
    df =  pd.read_csv('period_cycle_prediction\dataset\synthetic_data.csv', sep=',', header=0)
    # Preparando os dados para o modelo 
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

    # Modelo de Floresta Aleatória
    model_RD=RandomForestRegressor(criterion='squared_error', random_state=30, n_estimators=50)
    model_RD.fit(train_x, train_y)
    # Fazer a predição
    y_pred=model_RD.predict(test_x)
    output_pred = [[int(round(i[0])), int(round(i[1]))] for i in y_pred] # round the values 

    cycle_length=[]
    periods=[]
    for i in range(len(output_pred)):
        cycle_length.append(output_pred[i][0] )
        periods.append(output_pred[i][1] )

    # predição um passo a frente / novo ciclo
    predicao_um_passo_a_frente = model_RD.predict([test_x[-1]])
    cycles_numbers = np.arange(1, len(cycle_length)+1)

    plt.figure(figsize=(4,4))
    plt.rcParams.update({'font.size': 16})

    plt.plot(cycle_length, '-->', color='blue')
    plt.plot(periods, '-*', color='green')
    plt.plot(test_y, ':o', color='red')
    plt.plot(cycles_numbers[-1], predicao_um_passo_a_frente[0][0], '-->', color='blue')
    plt.plot(cycles_numbers[-1], predicao_um_passo_a_frente[0][1], '-*', color='green')
    plt.legend(['Duração dos ciclo (Predito)', 'Variação dos periodos (Predito)', 'Dados reais'])
    plt.grid()
    plt.xlabel('Ciclos')
    plt.ylabel('Dias')
    plt.title('Modelo Floresta Aleatória')
    plt.show()  

    # salvar figuras 
    plt.savefig('random.png', dpi=300, bbox_inches='tight')

    
    # calcular o RMSE 
    rms = sqrt(mean_squared_error(test_y, y_pred))
    print('RMSE: ', rms)

    # calcular o MAE (Mean Absolute Error)
    mae = np.mean(np.abs((test_y - y_pred)))
    print('MAE: ', mae)