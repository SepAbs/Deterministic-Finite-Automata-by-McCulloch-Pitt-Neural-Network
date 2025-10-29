from tensorflow import cond, constant, float32, multiply, reduce_sum
from warnings import filterwarnings
filterwarnings("ignore")
# Setting whatever achieved through calculations in 'Calculations' pdf file.
# Outputs = [S_1(t + 1), S_0(t + 1), y] # listInputs = [[S_1(t), S_0(t), x]] # listWeights = [[W_1, W_2, W_3] for i in range(number of McCulloch-Pitts neural networks = 3)]
Outputs, listInputs, listWeights, Threshold = ["S_1", "S_0", "y"], [constant([0, 0, 0], dtype = float32), constant([0, 0, 1], dtype = float32), constant([0, 1, 0], dtype = float32), constant([0, 1, 1], dtype = float32), constant([1, 0, 0], dtype = float32), constant([1, 0, 1], dtype = float32), constant([1, 1, 0], dtype = float32), constant([1, 1, 1], dtype = float32)], [constant([1, 1, -1], dtype = float32), constant([1.5, -0.5, 1.5], dtype = float32), constant([1, 0.5, -0.5], dtype = float32)], constant(1, dtype = float32)

# McCulloch-Pitts Neural Network
def McCulloch_Pitts_neuron(Inputs, Weights, Threshold):
    return cond(reduce_sum(multiply(Inputs, Weights)) >= Threshold, lambda:1.0, lambda:0.0)

def OneByOne(listInputs, listWeights, Threshold):
    for Inputs in listInputs:
        for Index, Weights in enumerate(listWeights):                                    
            print(f"\nFor inputs {Inputs} with weights {Weights} the output of {Index}th Mcculloh Pitts neural network returns {Outputs[Index]} is: {McCulloch_Pitts_neuron(Inputs, Weights, Threshold)}")
            
# Main Neural Network
def NeuralNetwork(listInputs, listWeights, Threshold):
    for Inputs in listInputs:
        print(f"\nFor inputs {Inputs} the neural network gives: {(McCulloch_Pitts_neuron(Inputs, listWeights[0], Threshold), McCulloch_Pitts_neuron(Inputs, listWeights[1], Threshold), McCulloch_Pitts_neuron(Inputs, listWeights[2], Threshold))}")

NeuralNetwork(listInputs, listWeights, Threshold)
