import numpy as np 
from os import path
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

X = np.array([  #training inputs
    [0,0],
    [0,1],
    [1,0],
    [1,1]])
y = np.array([  #training outputs
    [0],
    [1],
    [1],
    [0]])

if path.exists('xor.h5'): #open pretrained model if it exists
    model = load_model('xor.h5') 

else:
    model=Sequential()
    model.add(Dense(2, input_dim=2))    #2 hidden neurons and 2 input neurons
    model.add(Activation('tanh'))       #tanh as activation for hidden layer
    model.add(Dense(1))                 #1 output neuron
    model.add(Activation('tanh'))    #sigmoid as activation for output layer
    sgd = SGD(lr=0.45)    #Stochastic gradient descent               
    model.compile(loss='mse', optimizer=sgd)
    model.fit(X, y, batch_size=1, nb_epoch=1000)
    model.save('xor.h5')

print(model.predict_proba(X))
