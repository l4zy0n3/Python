import numpy as np
import time
class neuron:
    def sigmoid( self, x, deriv = False):
        return (np.tanh(x)) if not deriv else 1-np.tanh(x)*np.tanh(x)
        
    def __init__(self, inp, hid, out):
        np.random.seed( int(time.time()))
        self.inp = inp
        self.hid = hid
        self.out = out
        self.weights_ih = np.random.rand( self.hid, self.inp)
        self.weights_ho = np.random.rand( self.out, self.hid)
        self.bias_h = np.random.rand( self.hid, 1)
        self.bias_o = np.random.rand( self.out, 1)
        
    def guess( self, inputs):
        self.inputs = np.reshape( inputs ,[ len( inputs), 1])
        self.hidden = np.dot( self.weights_ih, self.inputs)
        self.hidden_values = self.sigmoid( self.hidden + self.bias_h)
        self.outputs =  np.dot( self.weights_ho, self.hidden_values)
        self.output_values = self.sigmoid( self.outputs + self.bias_o)
        return self.output_values
        
    def train( self, inputs, outputs):
        self.lr = 0.1
        for i in range(50000):
            index = np.random.randint(4)
            guess = self.guess( inputs[index])
            inp = np.reshape( inputs[index] ,[ len(inputs[index]), 1])
            target = outputs[index]

            #Calculate output gradients and deltas
            output_errors = ( target - guess)
            ho_gradients = np.multiply( self.sigmoid( self.outputs, deriv = True), output_errors)
            ho_gradients = np.multiply( ho_gradients, self.lr)
            ho_deltas = np.dot( ho_gradients, np.transpose(self.hidden_values))
            self.weights_ho += ho_deltas
            self.bias_o += ho_gradients
        
            #Calculate hidden gradients and deltas
            hidden_errors = np.dot( np.transpose(self.weights_ho), ho_gradients)
            ih_gradients = self.sigmoid( self.hidden_values, deriv = True)
            ih_gradients = np.multiply( ih_gradients, hidden_errors)
            ih_gradients = np.multiply( ih_gradients, self.lr)
            ih_deltas = np.dot( ih_gradients, np.transpose(inp))
            self.weights_ih += ih_deltas
            self.bias_h += ih_gradients
            
        print("Done")
