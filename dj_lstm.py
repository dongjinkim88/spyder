#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 20:46:56 2021

@author: djkim
"""

import random as rn
import numpy as np
import pandas as pd
import os
import glob
import datetime


df = pd.read_csv('cfa_nn_data.csv')

df.drop(['Unnamed: 0'],inplace=True,axis=1) #removing unnecessary columns

#reformatting and sorting by date
df['date'] = pd.to_datetime(df['date']) #always format the date
df = df.sort_values(['date'])

#Removing weekends
df = df[(df['date'].dt.dayofweek != 5)&(df['date'].dt.dayofweek != 6)]


#dropping if ptage of missing values is greater than 10%
df = df.loc[:, df.isna().sum()/df.shape[0] <= 0.1]
df = df.dropna(axis=0,how='all',subset=df.columns[1:])
df.head()


#linear interpolation to deal with missing data
date = df['date']
df = df[df.columns[1:]].astype(float).interpolate(method ='linear',axis = 0,limit=30,
                                                              limit_direction ='forward')

#sort by date
df['date'] = date
df = df.sort_values(['date'])


#Defining a function to calculate percentage change
def percentChange(x,numLags):
    '''
    INPUTS:
    x: Column for which we want to calculate percent change
    numLags: The number of days from when the change needs to be calculated. 
            Example : If using daily data - numLags = 1 for daily change
                                            numLags = 30 for monthly change
                                            numLags = 365 for yearly change       
    OUTPUT:
    percentage change in variable
    '''
    y = (x - x.shift(numLags))/x.shift(numLags)
    return y

dataForMl = pd.DataFrame()
dataForMl['date'] = df['date']

#here, I only have level variables so I do not need separate my variables into level vs non-level variables
levelVars = df.columns[:-1]
for levelVar in levelVars:
    dataForMl[f'{levelVar}Ret'] = percentChange(df[levelVar],1)
    
dataForMl = dataForMl[1:] #ignoring the first row as it contains null values



# Since we're going to forecast the one day ahead Nifty stock returns, the minimum lag considered by me is 1
minLagNum = 1

#lagging the vars :here i'm iginoring the ACF and PACF lag structure and deciding the maximum number of lags heuristically
maxLagNum = 10 #here I have chosen the maxLagNum arbitrarily. A better strategy is to look at the acf plot
dataForMl = dataForMl.sort_values(['date'])
for column in dataForMl.columns:
    for lag in range(minLagNum,maxLagNum+1):
        dataForMl[f'{column}Lag_{lag}'] = dataForMl[f'{column}'].shift(lag)
        

dataForMl.columns


#sort by date
dataForMl = dataForMl.sort_values(['date'])

#removing columns if nan value in a column
dataForMl = dataForMl.dropna()

#specifying independent variables:including only lagged versions of variables and excluding date variables
final_vars = [col for col in dataForMl.columns if (col.find('Lag')!=-1) & (col.find('date')==-1)]

#specifying the dependent variable
dep_var = 'National Stock Exchange: Index: Nifty 500Ret'

#always make the dependent ariable the last column in the dataset
final_vars.append(dep_var)

#for later use
dataForMl_copy = dataForMl

#keeping only relevant 
dataForMl = dataForMl[final_vars]





#breaking the data into train and test along time dim
test_percent = 0.10
no_test_obs =  int(np.round(test_percent*len(dataForMl)))
training = dataForMl[:-no_test_obs]
testing = dataForMl[-no_test_obs:]

#breaking the testing data into validation and out of sample data
validation_percent = 0.70
no_validation_obs = int(np.round(validation_percent*len(testing)))
validation = testing[:-no_validation_obs]
outOfSample = testing[-no_validation_obs:]

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
trainMinmax = min_max_scaler.fit_transform(training.values) #fit and transform training data
valMinmax = min_max_scaler.transform(validation.values)
outSampleMinmax = min_max_scaler.transform(outOfSample.values)


#breaking the data into independent variables (x) and dependent variables (y)

#training independent, dependent
trainMinmax_x,trainMinmax_y = trainMinmax[:,:-1],trainMinmax[:,-1] 

#validation independent, dependent
valMinmax_x,valMinmax_y = valMinmax[:,:-1],valMinmax[:,-1]

#out of sample testing independent, dependent
outSampleMinmax_x,outSampleMinmax_y = outSampleMinmax[:,:-1],outSampleMinmax[:,-1]





from numpy import array

#split a multivariate sequence into samples that preserve the temporal structure of the data
#SOURCE:https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


n_steps_in =30 #number of observations from the past that we assume to be relevant across time for forecasting
n_steps_out = 1 #number of units ahead that we want to forecast into the future

#training sequence
trainSeq_x, trainSeq_y = split_sequences(trainMinmax, n_steps_in,n_steps_out)

#out of sample sequence
validationSeq_x, validationSeq_y= split_sequences(valMinmax, n_steps_in,n_steps_out)

#out of sample sequence
outSampleSeq_x, outSampleSeq_y= split_sequences(outSampleMinmax, n_steps_in,n_steps_out)


trainMinmax.shape #Output: (rows,columns)
trainSeq_x.shape #Output: (number of samples,size of 'window' /timesteps,number of independent variables)


#make sure you have the correct versions installed
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)

# Implementing a neural network in Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, AveragePooling1D,MaxPooling1D
from tensorflow.keras.layers import Conv1D,AveragePooling1D,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1,l2,l1_l2


from tensorflow.keras.layers import LSTM

################################### Set for replicability ##################################################################
np.random.seed(42)
rn.seed(12345)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.random.set_seed(1234)
#from keras import backend as K
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
###########################################################################################################################


EarlyStop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto',restore_best_weights=True)
epochs = 100000
lr =0

sgd = SGD(lr=lr) #; adam = Adam(lr=lr) ;nadam = Nadam(lr=lr)
bs = 100

n_steps_in =30
n_steps_out = 1

#training sequence
trainSeq_x, trainSeq_y = split_sequences(trainMinmax, n_steps_in,n_steps_out)

#out of sample sequence
validationSeq_x, validationSeq_y= split_sequences(valMinmax, n_steps_in,n_steps_out)

#out of sample sequence
outSampleSeq_x, outSampleSeq_y= split_sequences(outSampleMinmax, n_steps_in,n_steps_out)

X_useless, y_useless = split_sequences(trainMinmax, n_steps_in,n_steps_out)
n_features = X_useless.shape[2]

np.random.seed(0); print(np.random.rand(4))

model = Sequential()
model.add(LSTM(300, #number of LSTM nodes
               input_shape=(n_steps_in, n_features),
               activation = 'tanh')) #ransformation:best to not use any other type of transformation
model.add(Dropout(0.1))
model.add(Dense(1,activation = 'linear'))
model.compile(loss='mean_squared_error', optimizer='sgd')

#model
model.fit(trainSeq_x, trainSeq_y,batch_size=bs,epochs=epochs, callbacks= [EarlyStop] ,verbose=2, shuffle=False,
                         validation_data =(validationSeq_x, validationSeq_y))



model.summary()

# validation metrics 
lstmValPred = model.predict(validationSeq_x)
#out of sample metrics
lstmOutSamplePred = model.predict(outSampleSeq_x)


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def plot_results(actual_y,predicted_y,method,date):
    mse = mean_squared_error(actual_y,predicted_y)
    
    plt.figure(figsize=(16,4))
    plt.plot(date,actual_y)
    plt.plot(date,predicted_y)
    plt.legend(['Actual','Predicted'])
    plt.title(f'{method} (MSE: {mse})')        
    plt.show()

plot_results(validationSeq_y ,lstmValPred  ,'LSTM Validation',range(len(validationSeq_y )))
plot_results(outSampleSeq_y ,lstmOutSamplePred  ,'LSTM Testing',range(len(outSampleSeq_y)))

