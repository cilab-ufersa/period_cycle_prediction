import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import generate_synthetic_data, generate_final_features, split_dataset, convet2dataframe
from darts import TimeSeries
from darts.models import AutoARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings 
warnings.filterwarnings("ignore")

if __name__ == '__main__': 

# load the data
total_regular_cycle_data = pd.read_csv('dataset/total_regular_cycle_data.csv')
features_total_regular_cycle_data, labels_total_regular_cycle_data = generate_final_features(total_regular_cycle_data)
input_train_total_regular_cycle, input_test_total_regular_cycle, output_train_total_regular_cycle, output_test_total_regular_cycle = split_dataset(features_total_regular_cycle_data, labels_total_regular_cycle_data, reshape=False)

input_train_total_regular_cycle_df = convet2dataframe(input_train_total_regular_cycle, ['period', 'cycle'])
output_train_total_regular_cycle = convet2dataframe(output_train_total_regular_cycle, ['period', 'cycle'])
input_test_total_regular_cycle_df = convet2dataframe(input_test_total_regular_cycle, ['period', 'cycle'])
series_test = TimeSeries.from_dataframe(input_test_total_regular_cycle_df, 'time', ['period'])
output_train_series = TimeSeries.from_dataframe(output_train_total_regular_cycle, 'time', ['period'])
series = TimeSeries.from_dataframe(input_train_total_regular_cycle_df, time_col='time', value_cols=['period'])

# series for cycle prediction
series_cycle = TimeSeries.from_dataframe(input_train_total_regular_cycle_df, time_col='time', value_cols=['cycle'])
series_cycle_test = TimeSeries.from_dataframe(input_test_total_regular_cycle_df, time_col='time', value_cols=['cycle'])

# train the model
model = AutoARIMA()
model.fit(series)

# make prediction
prediction_ = model.predict(len(series_test))
#-----------------------------------#
# model arima for cycle 
model_cycle = AutoARIMA()
model_cycle.fit(series_cycle)
# prediction the cycle 
prediction_cycle = model_cycle.predict(3)

testScore = np.sqrt(mean_squared_error(series_test.values(), prediction_.values()))
print('Test Score: %.2f MSE' % (testScore))
# calculate mean absolute error
testScore = mean_absolute_error(series_test.values(), prediction_.values())
print('Test Score: %.2f MAE' % (testScore))
# RMSE
print('Test Score: %.2f RMSE' % np.sqrt(testScore))
# calculate r2 score
testScore = r2_score(series_test.values(), prediction_.values())
print('Test Score: %.2f R2' % (testScore))

plt.figure(figsize=(4, 3))
plt.plot(np.arange(1,16),series_test.values()[-16:], '-->', linewidth=2.0)
plt.plot(np.arange(16, 17),prediction.values()[0].astype(int), 'o')
plt.plot(np.arange(16, 17),prediction.values()[0].astype(int), 'h')
plt.plot(np.arange(16, 17),prediction.values()[0].astype(int), '*')
# round the number in axis
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.ylabel('Days')
plt.xlabel('Months')
plt.legend(['Cycle serie', 'ARIMA', 'LSTM', 'Lasso'], loc='lower left')
plt.title('Case 1: Predicting the next cycle duration')
plt.grid(True)