#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     predict_bw_lstm1.py
# @author   Kyeong Soo (Joseph) Kim <Kyeongsoo.Kim@xjtlu.edu.cn>
# @date     2019-04-22
#           2022-03-23 - updated for TensorFlow version 2.6
#
# @brief    Predict channel bandwidth.
#
# @remarks  This code is based on the nice sample code from:
#           https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

# import modules
import numpy as np
import tensorflow as tf
import tensorflow.keras         # required for TF ver. 2.6
from skimage.util import view_as_windows

# define dataset
bws = np.load('bandwidths.npy')
X = view_as_windows(bws, 3, step=1)[:-1] # 3-sample sliding window over bws (except the last one, i.e., '[:-1]')
y = bws[3:]

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# define model
model = tf.keras.Sequential()
# model.add(tf.keras.layers.LSTM(units=50, activation='relu', input_shape=(3, 1)))
model.add(tf.keras.layers.LSTM(units=50, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction
for i in range(10):
    x_input = X[i]
    x_input = x_input.reshape((1, 3, 1))
    yhat = model.predict(x_input, verbose=0)
    print(f"{','.join([str(int(i)) for i in x_input.flatten()])} -> {yhat.flatten()[0]:.2e} (true value: {int(y[i]):d})")
