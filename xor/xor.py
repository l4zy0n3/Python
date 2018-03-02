from neuron import *
inputs = [ [ 0, 0],
           [ 0, 1],
           [ 1, 0],
           [ 1, 1] ]
outputs = [ 0,
            1,
            1,
            0 ]
nn = neuron( 2, 4, 1)
nn.train( inputs, outputs) 
print(nn.guess( [int(x) for x in input().split() ] ) )    
