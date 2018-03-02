import numpy as np
import time, pprint
from os import path
class NeuralNetwork:
    def sigmoid( self, x, deriv = False, alt = True):
        if alt:
            return (np.tanh(x)) if not deriv else 1-np.tanh(x)*np.tanh(x)
        return (1/(1+np.exp(x))) if not deriv else self.sigmoid(x)*(1-self.sigmoid(x))
    
    def __init__(self, inp, hid, out, lr = 0.25, epochs = 50000):
        #for consistent testing
        np.random.seed( int(time.time()))
        #init 
        self.inp = inp
        self.hid = hid
        self.out = out
        self.lr = lr
        self.epochs = epochs
        #load pretrained weights and biases
        self.weights_ih = np.load("weights_ih.npy") if path.isfile("weights_ih.npy") else np.random.rand( self.hid, self.inp)
        self.weights_ho = np.load("weights_ho.npy") if path.isfile("weights_ho.npy") else np.random.rand( self.out, self.hid)
        self.bias_h = np.load( "bias_h.npy") if path.isfile("bias_h.npy") else np.random.rand( self.hid, 1)
        self.bias_o = np.load( "bias_o.npy") if path.isfile("bias_o.npy") else np.random.rand( self.out, 1)
        

    def guess( self, inputs):
        self.inputs = np.reshape( inputs ,[ len( inputs), 1])
        self.hidden = np.dot( self.weights_ih, self.inputs)
        self.hidden_values = self.sigmoid( self.hidden + self.bias_h)
        self.output =  np.dot( self.weights_ho, self.hidden_values)
        self.output_values = self.sigmoid( self.output + self.bias_o)
        return self.output_values
        
    def train( self, inputs, outputs):
        for i in range( 1, self.epochs + 1):
            index = np.random.randint(4)
            guess = self.guess( inputs[index])
            inp = np.reshape( inputs[index] ,[ len(inputs[index]), 1])
            target = outputs[index]

            #Calculate output gradients and deltas
            output_errors = ( target - guess)
            ho_gradients = self.sigmoid( self.output, deriv = True)
            ho_gradients = np.multiply( ho_gradients, output_errors)
            ho_gradients = np.multiply( ho_gradients, self.lr)
            ho_deltas = np.dot( ho_gradients, np.transpose(self.hidden_values))
            self.weights_ho += ho_deltas
            self.bias_o += ho_gradients
        
            #Calculate hidden gradients and deltas
            hidden_errors = np.dot( np.transpose(self.weights_ho), ho_gradients)
            ih_gradients = self.sigmoid( self.hidden, deriv = True)
            ih_gradients = np.multiply( ih_gradients, hidden_errors)
            ih_gradients = np.multiply( ih_gradients, self.lr)
            ih_deltas = np.dot( ih_gradients, np.transpose(inp))
            self.weights_ih += ih_deltas
            self.bias_h += ih_gradients
            
            #printing status on the console
            if( i%10000 == 0):
                print( "******************************************************************************"\
                "\n\nEpoch #", i)
                print("\n\tInput -> Hidden Weights\n\t-----------------------\n")
                pprint.pprint( self.weights_ih)
                print("\n\tHidden Biases\n\t---------------\n")
                pprint.pprint( self.bias_h)
                print("\n\tHidden -> Outputs Weights\n\t-------------------------\n")
                pprint.pprint( self.weights_ho)
                print("\n\tOutput Biases\n\t---------------\n")
                pprint.pprint( self.bias_o)
                print("\n******************************************************************************\n")

            #saving weights and biases to storage
            if( i == self.epochs):
                np.save( "weights_ih", self.weights_ih)
                np.save( "weights_ho", self.weights_ho)
                np.save( "bias_h", self.bias_h)
                np.save( "bias_o", self.bias_o)

        print("Done")
