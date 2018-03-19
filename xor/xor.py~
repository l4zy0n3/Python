from neuron import NeuralNetwork
#delete npy files for retraining for another function
'''or_io = [[ [ 0, 0],
           [ 0, 1],
           [ 1, 0],
           [ 1, 1]],
           [ 0,
             1,
             1,
             1   ]]
and_io = [[[ 0, 0],
           [ 0, 1],
           [ 1, 0],
           [ 1, 1]],
           [ 0,
             0,
             0,
             1   ]]
'''
xor_io = [[[ 0, 0],
           [ 0, 1],
           [ 1, 0],
           [ 1, 1]],
           [ 0,
             1,
             1,
             0   ]]
nn = NeuralNetwork( 2, 4, 1)
#uncomment the next line to train
#nn.train( xor_io[0], xor_io[1]) 
print("XOR using MLP and Backprop...")
print( nn.guess( [int(x) for x in input("Inputs...\n").split() ])[0][0] )    
