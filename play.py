from Models.MLP.Layer import Layer
from Models.MLP.Network import Mlp
from Data import *

def test_layer_feed_with_size():
    ob = Layer(False)
    ob.init_with_size(5)
    print(ob)


def test_layer_feed_with_values():
    ob = Layer(True)
    ob.init_with_values([.90, 0.1, 0.5])
    print(ob)


def test_neuron_init_weights():
    ob = Layer(True, next_layer_size=3)
    ob.init_with_size(3)
    for i in ob.neurons:
        print(i)


def test_network_add_input_output_layer():
    mlp = Mlp(learning_rate=10, epochs=20000)
    mlp.lec_example1()
    mlp.fit(
        [
            [0, 0], [0, 1], [1, 0], [1, 1]
        ],
        [
            [0], [1], [1], [0]
        ]
    )
    # test_input = [[0,0],[0,1],[1,0],[1,1]]
    # test_output = [[0],[1],[1],[0]]
    # accuracy = 0
    # for i, j in zip(test_input, test_output):
    #     print(mlp.get_output(i), j)
    #     accuracy += int(mlp.get_output(i) == j)
    # accuracy /= len(test_input)
    # print(f"accuracy : {accuracy}")

def run_test():
    data = Data()
    train_input, train_output, test_input, test_output = data.GenerateData(['A', 'B', 'C'])
    mlp = Mlp(learning_rate=0.01, epochs=1000)
    # input_layer = Layer(False, 3)
    # input_layer.init_with_size(5)
    # hiddenlayer1 = Layer(True, 4)
    # hiddenlayer1.init_with_size(3)
    # hiddenlayer2 = Layer(True, 3)
    # hiddenlayer2.init_with_size(4)
    # output_layer = Layer(True, 0)
    # output_layer.init_with_size(3)
    # mlp.training_layers = [input_layer, hiddenlayer1, hiddenlayer2, output_layer]
    
    input_layer = Layer(False, 3)
    input_layer.init_with_size(5)
    hiddenlayer1 = Layer(True, 3)
    hiddenlayer1.init_with_size(3)
    output_layer = Layer(True, 0)
    output_layer.init_with_size(3)
    mlp.training_layers = [input_layer, hiddenlayer1, output_layer]

    mlp.fit(train_input, train_output)

    accuracy = 0
    for i, j in zip(test_input, test_output):
        accuracy += int(mlp.get_output(i) == j)
    accuracy /= len(test_input)
    print(f"accuracy : {accuracy}")

def do_work():
    run_test()
