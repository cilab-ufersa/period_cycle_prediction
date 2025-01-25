import sys
sys.path.append("../period_cycle_prediction/")

from period_cycle_prediction.utils import utils         
import numpy as np 
import pandas as pd
from joblib import load


if __name__ == '__main__':

    # Abrir histórico de dados 
    df =  pd.read_csv('period_cycle_prediction/dataset/synthetic_data.csv', sep=',', header=0)
    data_years = utils.calculate_cycle_and_periods(df)


    # load the model
    model_LR = load('linear_regression_model.joblib')

    # dados de teste para predição 
    # duracao_e_ciclo = np.array([[29,  5, 26,  6, 30,  6], #  cada linha contém informações de 3 ciclos e seus respectivos períodos. 
    #    [26,  5, 29,  6, 26,  5],
    #    [29,  6, 29,  6, 30,  6],
    #    [30,  6, 28,  6, 27,  6],
    #    [26,  5, 30,  5, 30,  6]])
    
    duracao_e_ciclo = np.array([[29,  5, 26,  6, 30,  6],
                                [26,  5, 29,  6, 26,  5]])
 
    
    # Fazer a predição
    y_pred = model_LR.predict(duracao_e_ciclo)
    output_pred = [[int(round(i[0])), int(round(i[1]))] for i in y_pred] # round the values 

    last_know_data_cycle = (data_years)[23] 
    
    predict_cycles_periods =  utils.next_period_prediction(last_know_data_cycle, np.array([y_pred]))



