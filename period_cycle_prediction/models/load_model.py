import sys
sys.path.append("../period_cycle_prediction/")

from period_cycle_prediction.utils import utils         
import numpy as np 
import pandas as pd
from joblib import load


if __name__ == '__main__':
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
 
    
    # Fazer a predição com os dados dos 6 ultimos ciclos
    y_pred = model_LR.predict(duracao_e_ciclo)
    
    last_know_data_cycle = ['2025-01-30', 30, 5]

    predict_cycles_periods =  utils.next_period_prediction(last_know_data_cycle, np.array([y_pred]))



