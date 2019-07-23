import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from matplotlib import pyplot as plt
import json
import time

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
    def __init__(self, filename="model", iosize=[1,1], learn_rate=0.2):
        self.layers = [] # Layers containing the persistent weights/biases
        self.weights = []
        self.biases = []
        self.output_layer = None
        self.set_learn_rate(learn_rate)
        self.iosize = iosize
        self.model_accuracy = 0.0
        self.filename = filename
    def set_learn_rate(self, learn_rate):
        self.learn_rate = learn_rate

    # Adds a layer to the model
    def add_layer(self, size, weights=[], biases=[], activation='relu'):
        # print("Adding layer of size: " + str(size))
        # print("Layer activation: " + activation)
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
        tf.compat.v1.reset_default_graph()
        # 1 dimensional array placeholder for our input and output data.
        x = tf.compat.v1.placeholder(tf.float32, [None, self.iosize[0]], name='x')
        y = tf.compat.v1.placeholder(tf.float32, [None, self.iosize[1]], name='y')
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

        # Last thing in our initialization is to save the initialization operation to use when we run the graph later.
        self.init_op = tf.global_variables_initializer()

    def store_weights_biases(self, sess):
        # Store the weights in our layers to reload for more training or predictions.
        for layer in range(len(self.layers)):
            w = self.weights[layer].eval()
            b = self.biases[layer].eval()
            self.layers[layer].set_weights(np.asarray(self.weights[layer].eval()))
            self.layers[layer].set_bias(np.asarray(self.biases[layer].eval()))
            self.output_layer.set_weights(np.asarray(self.weights[-1].eval()))
            self.output_layer.set_bias(np.asarray(self.biases[-1].eval()))

    def train(self, data, labels, epochs=3, test_data=[], test_labels=[], x_val=[], y_val=[], verbose=False):
        with tf.Session() as sess:
            sess.run(self.init_op)
            for epoch in range(epochs):
                _, c = sess.run([self.optimizer, self.cross_entropy], feed_dict={self.x: data, self.y: labels})
                if(verbose):
                    print("Epoch " + str(epoch) + " cost: " + str(c))
            if(len(test_data)>0):
                self.model_accuracy = sess.run(self.accuracy, feed_dict={self.x: test_data, self.y: test_labels})
            self.store_weights_biases(sess)
            if(verbose and len(test_data)>0):
                print("Accuracy on the test set: " + str(self.model_accuracy))

    def accuracy_check(self, data, labels, store=False):
        self.init_graph()
        with tf.Session() as sess:
            sess.run(self.init_op)
            accuracy = sess.run(self.accuracy, feed_dict={self.x: data, self.y: labels})
            if(store):
                self.model_accuracy = accuracy
            return accuracy
        
    # Static method to create a new model from a filename.
    def load_from_file(filename):
        print("Loading model from filename: " + filename)
        json_model = json.load(open(filename, "r"))
        model = Model(filename = filename, iosize=json_model['iosize'], learn_rate=json_model['learn_rate'])

        layers = json_model['layers']
        for layer in layers:
            model.add_layer(size=layer['size'],weights=layer['weights'],biases=layer['bias'], activation=layer['activation'])
    
        return model
    def clear_model(self):
        del self.layers

    def save_model(self, filename=""):
        # We need to save....
        # Layers
        # - Weights
        # - Biases
        # - Activations
        # Model
        # - learn rate
        # - cost func
        # - IO size wv
        # print("Saving model to filename: " + filename)
        model = { 'iosize' : self.iosize,
                  'learn_rate' : self.learn_rate,
                  'layers' : None,
                  'accuracy' : str(self.model_accuracy)
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
        if(len(filename)==0):
            json.dump(model, open(self.filename,'w'), sort_keys=True, indent=4)
        else:
            json.dump(model, open(filename,'w'), sort_keys=True, indent=4)

    def predict(self, data, sess=None):
        if(sess==None):
            self.init_graph()
            with tf.Session() as sess:
                sess.run(self.init_op)
                return sess.run(self.output, feed_dict={self.x: data})
        else:
            return sess.run(self.output, feed_dict={self.x: data})
