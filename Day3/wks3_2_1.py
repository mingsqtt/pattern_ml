#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:00:55 2017

@author: NPBME
"""

import h5py
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


                            # Set up 'ggplot' style
plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'


# Retrieve data  --------------------------------------------------------------
#
#
#

seed = 7
np.random.seed(seed)

f                   = h5py.File('cad5sec.mat')
X                   = f["data"]
Y                   = f["classLabel"]
data                = np.array(X)
data                = np.transpose(data)
label               = np.array(Y)
label               = np.transpose(label)


nor                 = data[0:32000]
cad                 = data[32000:38120]



# stepzation function ---------------------------------------------------------
#
#
# This function creates segmenets of a 1D signal
# It works in batch
#
#
# Dependency:       numpy

def makeSteps(dat, length, dist):
    width           = dat.shape[1]
    numOfSteps      = int(np.floor((width-length)/dist)+1)
    
                                        # Initialize the output
    segments        = np.zeros([dat.shape[0],numOfSteps,length],
                               dtype=dat.dtype)
    
    for l in range(numOfSteps):
        segments[:,l,:]     = dat[:,(l*dist):(l*dist+length)]
        
    return segments





# Splitting data into training and testing set---------------------------------
#
#
#


print('Create dataset...')
trNor               = nor[0:28800].copy()
tsNor               = nor[28800:32000].copy()
trCad               = cad[0:5000].copy()
tsCad               = cad[5000:6120].copy()




# Create segments from the signals --------------------------------------------
#
#
#

length              = 36
dist                = 24

print('Finalizing all the data ....')
trNorS              = makeSteps(trNor, length, dist)
tsNorS              = makeSteps(tsNor, length, dist)
trCadS              = makeSteps(trCad, length, dist)
tsCadS              = makeSteps(tsCad, length, dist)

trDat               = np.vstack([trNorS,trCadS])
tsDat               = np.vstack([tsNorS,tsCadS])

trLbl               = np.vstack([np.zeros([trNorS.shape[0],1]),
                                 np.ones([trCadS.shape[0],1])])
tsLbl               = np.vstack([np.zeros([tsNorS.shape[0],1]),
                                 np.ones([tsCadS.shape[0],1])])






# Creating model --------------------------------------------------------------
#
#
#


print('Creating lstm model...')

modelname   = 'wks3_2_1'

def createModel():    
#    model          = Sequential()
#    model.add(LSTM(32, input_shape=(trDat.shape[1], length), 
#                   return_sequences=True, 
#                   dropout=0.25, 
#                   recurrent_dropout=0.25)) 
#    model.add(LSTM(32, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))             
#    model.add(LSTM(32, return_sequences=True, dropout=0.25))
#    model.add(LSTM(64, dropout=0.25))                                     
#    model.add(Dense(1, activation='sigmoid'))
    
    inputs      = Input(shape=(trDat.shape[1],length))
    y           = LSTM(32, 
                       return_sequences=True, 
                       dropout=0.25, 
                       recurrent_dropout=0.25)(inputs)
    y           = LSTM(32, 
                       return_sequences=True, 
                       dropout=0.5, 
                       recurrent_dropout=0.25)(y)
    y           = LSTM(32, 
                       return_sequences=True, 
                       dropout=0.25)(y)
    y           = LSTM(64, 
                       dropout=0.25)(y)
    y           = Dense(1, activation='sigmoid')(y)
    
    model       = Model(inputs=inputs,outputs=y)
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

    return model





                            # Setup the models
model       = createModel() # This is meant for training
modelGo     = createModel() # This is used for final testing

model.summary()

# .............................................................................


                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy
filepath        = modelname + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger]



# .............................................................................


                            # Fit the model
                            # This is where the training starts
model.fit(trDat, 
          trLbl, 
          validation_data=(tsDat, tsLbl), 
          epochs=40, 
          batch_size=128,
          shuffle=True,
          callbacks=callbacks_list)



# ......................................................................


                            # Now the training is complete, we get
                            # another object to load the weights
                            # compile it, so that we can do 
                            # final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])

 



# .......................................................................


                            # Make classification on the test dataset
predicts    = modelGo.predict(tsDat)

labelname   = ['Normal','CAD']
                                            # the labels for the classfication report


testScores  = metrics.accuracy_score(tsLbl,predicts.round())
confusion   = metrics.confusion_matrix(tsLbl,predicts.round())


print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(tsLbl,predicts.round(),target_names=labelname,digits=4))
print(confusion)


    
    
    
# ..................................................................
    
import pandas as pd

records     = pd.read_csv(modelname +'.csv')
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.yticks([0.00,0.40,0.60,0.80])
plt.title('Loss value',fontsize=12)

ax          = plt.gca()
ax.set_xticklabels([])



plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.yticks([0.6,0.7,0.8,0.9])
plt.title('Accuracy',fontsize=12)
plt.show()




# ................................................................

from tensorflow.keras.utils import plot_model

plot_model(model, 
           to_file=modelname+'_model.pdf', 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')