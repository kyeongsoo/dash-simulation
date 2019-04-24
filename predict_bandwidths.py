#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     predict_bandwidths.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2019-04-22
#
# @brief    A function predicting future bandwidths based on past history.
#
# @remarks  The current implementation is based on the trained model from
#           'predict_bw_lstm2.py'.

### import modules
import numpy as np
from keras.models import load_model
from sklearn.externals import joblib


def predict_bandwidths(history, num_future_segments=1):
    """
    Predict future bandwidths up to num_future_segments based on history.
    N.B.: if num_future_segments is greater than one, you need to modify
    'predict_bw_lstm2.py' accordingly.
    """
    # load saved model and scaler from 'predict_bw_lstm2.py'
    model = load_model('lstm_model.h5')
    scaler = joblib.load('lstm_scaler.joblib')

    # scale input data
    history = history.reshape((len(history), 1))  # [n_samples, n_features] for scaler
    history = scaler.fit_transform(history)

    # reshape to [samples, time steps, features]
    history = np.reshape(history, (history.shape[0], 1, history.shape[1]))
    
    bws = model.predict(history)         

    # invert prediction
    bws = scaler.inverse_transform(bws)
    
    return bws.flatten()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    bws = np.load('bandwidths.npy')
    bws = bws[:30]              # for test
    predicted = np.zeros(len(bws))
    predicted[0] = np.nan

    for i in range(len(bws)-1):
        predicted[i+1] = predict_bandwidths(np.array([bws[i]]))

    plt.close('all')
    plt.plot(bws, label='True')
    plt.plot(predicted, label='Predicted')
    plt.xlabel('Segment Number')
    plt.ylabel('Bandwidth [kbps]')
    plt.legend(loc=1)
    plt.show()
