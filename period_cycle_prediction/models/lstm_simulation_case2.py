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

    regular_cycle_data = pd.read_csv('dataset/regular_cycle_data.csv')
    features_regular_cycle_data, labels_regular_cycle_data = generate_final_features(regular_cycle_data)
    input_train_regular_cycle, input_test_regular_cycle, output_train_regular_cycle, output_test_regular_cycle = split_dataset(features_regular_cycle_data, labels_regular_cycle_data, reshape=False)

    n_features = input_train_regular_cycle.shape[2]
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(input_train_regular_cycle.shape[1], input_train_regular_cycle.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(units=n_features, activation='relu'))

    opt=tf.keras.optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer=opt)
    # add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(input_train_regular_cycle, output_train_regular_cycle, epochs=2000, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # plot the loss and val loss
    plt.figure(figsize=(4, 3))
    plt.plot(history.history['loss'], '-', linewidth=2)
    plt.plot(history.history['val_loss'], '--', linewidth=2)
    plt.grid(True)
    plt.legend(['Train', 'Validation'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Case 2: LSTM model loss')
    # log scale
    #plt.yscale('log')

    # add a zoom in epoch 70 to 100
    ax = plt.axes([0.6, 0.4, .20, .20])
    plt.plot(history.history['loss'], '-', linewidth=2)
    plt.plot(history.history['val_loss'], '--', linewidth=2)
    plt.grid(True)
    ax.set_ylim(1, 6)
    ax.set_xlim(1500, 1650)


    # save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('case2_history_lstm.csv', index=False)

    # save figure
    fig = plt.gcf()
    fig.savefig('case2_loss_lstm.eps', dpi=300, bbox_inches='tight')

    # save model 
    model.save('case2_lstm_model.h5')
