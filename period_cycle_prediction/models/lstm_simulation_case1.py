import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from utils.utils import generate_synthetic_data, generate_final_features, split_dataset
import warnings 
warnings.filterwarnings("ignore")

if __name__ == '__main__': 

    total_regular_cycle_data = pd.read_csv('dataset\\total_regular_cycle_data.csv')
    features_total_regular_cycle_data, labels_total_regular_cycle_data = generate_final_features(total_regular_cycle_data)
    input_train_total_regular_cycle, input_test_total_regular_cycle, output_train_total_regular_cycle, output_test_total_regular_cycle = split_dataset(features_total_regular_cycle_data, labels_total_regular_cycle_data, reshape=False)

    # create and fit the LSTM network
    n_features = input_train_total_regular_cycle.shape[2]
    model = Sequential()
    model.add(LSTM(64, input_shape=(3, n_features),  activation='tanh'))
    model.add(Dropout(0.05))
    model.add(Dense(n_features, activation='relu'))
    model.summary()

    opt=tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=opt, run_eagerly=True)
    # add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(input_train_total_regular_cycle, output_train_total_regular_cycle, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

    # plot the loss and val loss
    plt.figure(figsize=(4, 3))
    plt.plot(history.history['loss'], '-', linewidth=2)
    plt.plot(history.history['val_loss'], '--', linewidth=2)
    plt.grid(True)
    plt.legend(['Train', 'Validation'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Case 1: LSTM model loss')
    ax = plt.axes([0.6, 0.4, .20, .20])
    plt.plot(history.history['loss'], '-', linewidth=2)
    plt.plot(history.history['val_loss'], '--', linewidth=2)
    plt.grid(True)
    ax.set_ylim(0.1, 3)
    ax.set_xlim(70, 93)
    # save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('case1_history_lstm.csv', index=False)
