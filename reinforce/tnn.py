#!/usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from matplotlib import pyplot as plt
import json

# Redoing custom neural network with tensorflow to avoid the input between layers from blowing up.
# Going to use eager execution for this for quicker prototyping. 
# tfe.enable_eager_execution()


# Layer class, contains the weights coming into the layer and the biases of the layer.
class Layer:
    def __init__(self, size, weights=[], bias=[], activation='relu', hidden=True):
        assert (len(size) == 2), "Size must be a tuple of size 2."
        self.size = size
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def set_hidden(self, hidden):
        self.hidden = hidden

    def get_hidden(self):
        return self.hidden

    def set_activation(self, activation):
        self.activation = activation

    def set_weights(self, weights):
        self.weights = weights

    def set_bias(self, bias):
        self.bias = bias

    def set_size(self, size):
        self.size = size

    def get_activation(self):
        return self.activation

    def get_bias(self):
        return self.bias

    def get_weights(self):
        return self.weights

    def get_size(self):
        return self.size

class Model:
    # initialize stuff.
    # inputs
    # size - input/output layer sizes
    # filename - If supplied, then we load data from file.
    def __init__(self, filename=None, iosize=[1,1], learn_rate=0.2):
        self.layers = [] # Layers containing the persistent weights/biases
        self.weights = []
        self.biases = []
        self.output_layer = None
        self.set_learn_rate(learn_rate)
        self.iosize = iosize

    def set_learn_rate(self, learn_rate):
        self.learn_rate = learn_rate

    # Adds a layer to the model
    def add_layer(self, size, weights=[], biases=[], activation='relu'):
        print("Adding layer of size: " + str(size))
        print("Layer activation: " + activation)
        if(self.output_layer==None):
            self.output_layer = Layer(size, weights, biases, activation)
        else:
            self.layers.append(self.output_layer)
            self.output_layer = Layer(size, weights, biases, activation)

        # self.weights.append(tf.Variable(tf.random_normal(size, stddev=0.03, dtype='float64')))
        # self.biases.append(tf.Variable(tf.random_normal([size[1]], dtype='float64')))

    # initializes the graph based on the current layers, activation functions, cost function, etc. 
    def init_graph(self):
        # First we clear the graph
        tf.reset_default_graph()
        # 1 dimensional array placeholder for our input and output data.
        x = tf.placeholder(tf.float32, [None, self.iosize[0]], name='x')
        y = tf.placeholder(tf.float32, [None, self.iosize[1]], name='y')
        self.x = x
        self.y = y

        # TODO - initialize layers based on model data
        assert (len(self.layers) > 0), "Must have 1 or more hidden layers"
        biases = []
        weights = []
        # Load layer data
        for layer in range(len(self.layers)):
            l = self.layers[layer]
            if(len(l.get_weights())==0):
                # Initialize weights/biases randomly
                # print("Generating random layer of size: " + str(l.get_size()))
                weights.append(tf.Variable(tf.random_normal(l.get_size(), stddev=0.03)))
                biases.append(tf.Variable(tf.random_normal([l.get_size()[1]])))

            else:
                # initialize based on the layer data
                weights.append(tf.Variable(l.get_weights()))
                biases.append(tf.Variable(l.get_bias()))

        # Setup output layer variables
        if(len(self.output_layer.get_weights())==0):
            weights.append(tf.Variable(tf.random_normal(self.output_layer.get_size(), stddev=0.03)))
            biases.append(tf.Variable(tf.random_normal([self.output_layer.get_size()[1]])))
        else:
            weights.append(tf.Variable(self.output_layer.get_weights()))
            biases.append(tf.Variable(self.output_layer.get_bias()))
        
        self.weights = weights
        self.biases = biases

        # Setup feed forward process
        for layer in range(len(self.layers)):
            if(layer==0):
                hidden_out = tf.add(tf.matmul(x, weights[0]), biases[0])
            else:
                hidden_out = tf.add(tf.matmul(hidden_out, weights[layer]), biases[layer])

            if(self.layers[layer].get_activation()=='relu'):
                hidden_out = tf.nn.relu(hidden_out)
            elif(self.layers[layer].get_activation()=='tanh'):
                hidden_out = tf.nn.tanh(hidden_out)
            else:
                print("Encountered shenanigans")
                exit(1)

        # print("Activation of output: " + self.output_layer.get_activation())
        if(self.output_layer.get_activation()=='softmax'):
            y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, weights[-1]), biases[-1]))
        elif(self.output_layer.get_activation()=='relu'):
            print("Implement this you derp")
            exit(1)
        elif(self.output_layer.get_activation()=='tanh'):
            y_ = tf.nn.tanh(tf.add(tf.matmul(hidden_out, weights[-1]), biases[-1]))

        # Our output function used for predictions later.
        self.output = y_
        # cost function
        y_clipped = tf.clip_by_value(y_, 1e-10, 0.99999999)
        # cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1-y) * tf.log(1 - y_clipped), axis=1))
        cross_entropy = tf.losses.mean_squared_error(y, y_clipped)
        self.cross_entropy = cross_entropy
        # optimizer
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate).minimize(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(cross_entropy)
        self.optimizer = optimizer

        # Accuracy assessment
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy = accuracy
        # # Add it to the summary
        # tf.summary.scalar('accuracy', accuracy)

        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter('./')

        # Last thing in our initialization is to save the initialization operation to use when we run the graph later.
        self.init_op = tf.global_variables_initializer()

    def train(self, data, labels, epochs=3, test_data=[], test_labels=[], verbose=False):
        with tf.Session() as sess:
            sess.run(self.init_op)
            for epoch in range(epochs):
                _, c = sess.run([self.optimizer, self.cross_entropy], feed_dict={self.x: data, self.y: labels})
                if(verbose):
                    print("Epoch " + str(epoch) + " cost: " + str(c))
            if(len(test_data)>0):
                print("Accuracy after training")
                print(sess.run(self.accuracy, feed_dict={self.x: test_data, self.y: test_labels}))

            # Store the weights in our layers to reload for more training or predictions.
            for layer in range(len(self.layers)):
                w = self.weights[layer].eval()
                b = self.biases[layer].eval()
                self.layers[layer].set_weights(np.asarray(self.weights[layer].eval()))
                self.layers[layer].set_bias(np.asarray(self.biases[layer].eval()))
            self.output_layer.set_weights(np.asarray(self.weights[-1].eval()))
            self.output_layer.set_bias(np.asarray(self.biases[-1].eval()))

    def accuracy_check(self, data, labels):
        self.init_graph()
        with tf.Session() as sess:
            sess.run(self.init_op)
            print("Accuracy")
            print(sess.run(self.accuracy, feed_dict={self.x: data, self.y: labels}))
    # Static method to create a new model from a filename.
    def load_from_file(filename):
        print("Loading model from filename: " + filename)
        json_model = json.load(open(filename, "r"))
        model = Model(iosize=json_model['iosize'], learn_rate=json_model['learn_rate'])

        layers = json_model['layers']
        for layer in layers:
            model.add_layer(size=layer['size'],weights=layer['weights'],biases=layer['bias'], activation=layer['activation'])
    
        return model
    def save_model(self, filename="model"):
        # We need to save....
        # Layers
        # - Weights
        # - Biases
        # - Activations
        # Model
        # - learn rate
        # - cost func
        # - IO size wv
        print("Saving model to filename: " + filename)
        model = { 'iosize' : self.iosize,
                  'learn_rate' : self.learn_rate,
                  'layers' : None
        }
        layer_num = 0
        json_layers = []
        for layer in self.layers:
            weights = layer.get_weights().tolist()
            biases = layer.get_bias().tolist()
            l = { '_layer_num' : layer_num,
                  'size' : layer.get_size(),
                  'activation' : layer.get_activation(),
                  'weights' : weights,
                  'bias' : biases
            }
            json_layers.append(l)
            layer_num+=1
        weights = self.output_layer.get_weights().tolist()
        biases = self.output_layer.get_bias().tolist()
        l = { '_layer_num' : layer_num,
              'size' : self.output_layer.get_size(),
              'activation' : self.output_layer.get_activation(),
              'weights' : weights,
              'bias' : biases
        }
        json_layers.append(l)
        model['layers'] = json_layers
        json.dump(model, open(filename,'w'), sort_keys=True, indent=4)

    def predict(self, data, sess=None):
        if(sess==None):
            self.init_graph()
            with tf.Session() as sess:
                sess.run(self.init_op)
                return sess.run(self.output, feed_dict={self.x: data})
        else:
            return sess.run(self.output, feed_dict={self.x: data})


    # multiplication unit test
    def mult_test(self):
        # Weight matrix
        w = np.array([0.1, 0.2, 0.3, 0.4])
        w=w.reshape(2,2)
        w = tf.Variable(w,name='w')
        # x input data. 
        x = tf.Variable([1.,1.], dtype='float64')
        # Needs to be expanded to a rank 2 tensor.
        x = tf.expand_dims(x,0)
        print("x " + str(x.numpy()))
        print("w " + str(w.numpy()))
        print("matmul")
        c = tf.matmul(x,w)
        print(c)

def load_mnist_test():
    model = Model.load_from_file("mnist100.m")

    from tensorflow.keras.datasets import mnist
    (x_dataset, y_labels), (x_test, y_test) = mnist.load_data()
    print("Normalizing the dataset")
    x_dataset = x_dataset/255.0
    print("Reshaping dataset")
    x_dataset = x_dataset.reshape(60000, 784)
    print("Input dataset size: " + str(x_dataset.shape))
    # Create one hots for labels
    labels = []
    # print("First label: " + str(y_labels[0]))
    for label in y_labels:
        labels.append(np.zeros(10))
        labels[-1][label] = 1
        
    test_labels = []
    for label in y_test:
        test_labels.append(np.zeros(10))
        test_labels[-1][label] = 1
    test_labels = np.array(test_labels)
    y_labels = np.array(labels)

    x_test = x_test / 255.0
    print("x_test size: " + str(x_test.shape))
    x_test = x_test.reshape(10000, 784)
    print("Labels dataset size: " + str(y_labels.shape))

    print("Accuracy check")
    model.accuracy_check(x_test, test_labels)
    np.set_printoptions(suppress=True)
    prediction = model.predict(x_test[:5])
    print(prediction)
    print("Labels")
    print(test_labels[:5])


def load_mnist_data():
    from tensorflow.keras.datasets import mnist
    
    (x_dataset, y_labels), (x_test, y_test) = mnist.load_data()

    print("Normalizing the dataset")
    x_dataset = x_dataset/255.0
    print("Reshaping dataset")
    x_dataset = x_dataset.reshape(60000, 784)
    print("Input dataset size: " + str(x_dataset.shape))
    # Create one hots for labels
    labels = []
    # print("First label: " + str(y_labels[0]))
    for label in y_labels:
        labels.append(np.zeros(10))
        labels[-1][label] = 1
        
    test_labels = []
    for label in y_test:
        test_labels.append(np.zeros(10))
        test_labels[-1][label] = 1
    test_labels = np.array(test_labels)
    y_labels = np.array(labels)

    x_test = x_test / 255.0
    print("x_test size: " + str(x_test.shape))
    x_test = x_test.reshape(10000, 784)

    return (x_dataset, y_labels), (x_test, test_labels)

def mnist_test():
    model = Model(iosize=[784, 10])
    model.set_learn_rate(0.0005)
    model.add_layer([784, 392])
    model.add_layer([392, 196])
    model.add_layer([196,10], activation='tanh')

    model.init_graph()
    filename = "mnist_model_1"

    epochs=500

    (x_dataset, y_labels), (x_test, test_labels) = load_mnist_data()
    # x_dataset = x_dataset[:1000] # First 30k
    # y_labels = y_labels[:1000]
    # x_dataset = x_dataset[15000:] # Second 30k
    # y_labels = y_labels[15000:]
    model.train(x_dataset, y_labels, epochs=epochs, test_data=x_test, test_labels=test_labels, verbose=True)
    print("Post training")
    model.accuracy_check(x_test, test_labels)
    np.set_printoptions(suppress=True)
    prediction = model.predict(x_test[:5])
    print(prediction)
    print("Labels")
    print(test_labels[:5])
    model.save_model(filename)
    np.set_printoptions(suppress=True)
    with tf.Session() as sess:
        sess.run(model.init_op)
        print("First label: " + str(test_labels[0]))
        print("Second label: " + str(test_labels[1]))
        print("Third label: " + str(test_labels[2]))
        test = sess.run(model.output, feed_dict={model.x: x_test[:3]})
        print(test)
        print(sess.run(model.output(feed_dict={model.x: x_dataset[0], model.y: y_labels[0]})))


def ttt_train_model():
    import reader
    print("Loading tic tac toe database")
    (x_states, x_labels), (o_states, o_labels) = reader.load_one_hot_db("ttt.db")
    print("Shuffling data")
    import random
    # Shuffle x_states & x_labels together
    x_data = list(zip(x_states,x_labels))
    random.shuffle(x_data)
    x_states, x_labels = zip(*x_data)

    o_data = list(zip(o_states, o_labels))
    random.shuffle(o_data)
    o_states, o_labels = zip(*o_data)

    print("Converting to numpy arrays")
    x_states = np.array(x_states)
    x_labels = np.array(x_labels)
    o_states = np.array(o_states)
    o_labels = np.array(o_labels)

    print("Creating accuracy test data")
    ac_num = 1000 # test data number of entries
    x_test_states = x_states[-ac_num:]
    x_test_labels = x_labels[-ac_num:]
    x_states = x_states[:-ac_num]
    x_labels = x_labels[:-ac_num]

    o_test_states = o_states[-ac_num:]
    o_test_labels = o_labels[-ac_num:]
    o_states = o_states[:-ac_num]
    o_labels = o_labels[:-ac_num]

    # Building the model
    model = Model(iosize=[18,9])
    model.add_layer([18,72], activation='tanh')
    model.add_layer([72,72], activation='tanh')
    model.add_layer([72,9], activation='softmax')
    model.set_learn_rate(0.03)
    model.init_graph()
    print(o_states[0])
    print(o_labels[0])
    # Train teh model to play O
    model.train(o_states, o_labels, epochs=10000, test_data=o_test_states, test_labels=o_test_labels, verbose=True)
    # print("Prediction with game states")
    # print(x_test_states[:2])
    # print("Board 1")
    # print_ttt_board(x_test_states[0])
    # print("Board 2")
    # print_ttt_board(x_test_states[1])

    # prediction = model.predict(x_test_states[:2])
    # p1 = prediction[0]
    # p2 = prediction[1]
    # p1 = np.argmax(p1)
    # p2 = np.argmax(p2)
    # np.set_printoptions(suppress=True)
    # print(prediction)
    # print("Some labels to go with that")
    # print(x_test_labels[:2])
    # print("Board 1 with prediction applied")
    # print_ttt_board(x_test_states[0], p1)
    # print("Board 2 with prediction applied")
    # print_ttt_board(x_test_states[1], p2)

    model.save_model("ttt_model")

def print_ttt_board(state, move=None):
    board = ['.','.','.',
             '.','.','.',
             '.','.','.']
    for i in range(9):
        if(state[i]==1):
            # x exists
            board[i] = 'X'
        elif(state[i+9]==1):
            # o exists
            board[i] = 'O'
        else:
            # nothing there
            pass
    if(move!=None):
        print("applying move to board: " + str(move))
        if(move<9):
            board[move]='O'
        else:
            print("Invalid move")
    board = np.array(board)
    board=board.reshape(3,3)
    print(board[0])
    print(board[1])
    print(board[2])

def ttt_valid_move(game_state, move):
    if(move<0 or move>17):
        return False
    if(move<=8):
        if(game_state[move]==0 and game_state[move+9]==0):
            return True
        else:
            return False
    else:
        if(game_state[move]==0 and game_state[move-9]==0):
            return True
        else:
            return False

# Returns the sorted list of moves suggested by the engine.
def ttt_order_prediction(prediction):
    moves = []
    while(len(prediction)!=0):
        p = np.argmax(prediction)
        prediction = np.delete(prediction, p) # remove the index at p
        moves.append(p)
    return moves

# Returns the first valid move given a game state and a list of moves
def get_first_valid_move(game_state, moves):
    print("Fetching valid move ")
    print(game_state)
    print(moves)
    for move in moves:
        if(ttt_valid_move(game_state, move)):
            return move

def interactive_tic_tac_toe():
    model = Model.load_from_file("ttt_model")

    game_state = np.zeros(18)
    print_ttt_board(game_state)
    game_state = game_state.reshape(1,18)
    model.init_graph()
    with tf.Session() as sess:
        sess.run(model.init_op)
        print_ttt_board(game_state[0])
        print("Computer will go first as O")
        prediction = model.predict(game_state, sess)
        p = ttt_order_prediction(prediction)
        computer_move = get_first_valid_move(game_state[0], p)
        game_state[0][computer_move+9]=1
        print_ttt_board(game_state[0])
        while(True):
            # Getting user input
            move = input("Type a number 1-9 for the index to place your X: ")
            move = int(move)

            while(ttt_valid_move(game_state[0], move)==False):
                print("Invalid move input")
                move = input("Type a number 1-9 for the index to place your X: ")
                move = int(move)
            game_state[0][move]=1
            print_ttt_board(game_state[0])

            np.set_printoptions(suppress=True)
            print("Computer move")
            prediction = model.predict(game_state, sess)
            print("Prediction vector")
            print(prediction)
            p = ttt_order_prediction(prediction)
            print("Prediction array: " + str(p))
            c_move = get_first_valid_move(game_state[0],p)
            game_state[0][c_move+9]=1
            print_ttt_board(game_state[0])
# mnist_test()
# load_mnist_test()
# ttt_train_model()
interactive_tic_tac_toe()
# ttt_test()

# from tensorflow.keras.datasets import mnist
# (x_dataset, y_labels), (x_test, y_test) = mnist.load_data()
# print("Normalizing the dataset")
# x_dataset = x_dataset/255.0
# print("Reshaping dataset")
# x_dataset = x_dataset.reshape(60000, 784)
# ins, outs = model.feedforward(x_dataset[1])
# print(outs[-1])

