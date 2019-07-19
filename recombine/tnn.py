#!/usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from matplotlib import pyplot as plt
import json

# Redoing custom neural network with tensorflow to avoid the input between layers from blowing up.
# Going to use eager execution for this for quicker prototyping. 
# tfe.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
        self.model_accuracy = 0.0
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
            acc = sess.run(self.accuracy, feed_dict={self.x: data, self.y: labels})
            print(acc)
            return acc
    # Static method to create a new model from a filename.
    def load_from_file(filename):
        print("Loading model from filename: " + filename)
        json_model = json.load(open(filename, "r"))
        model = Model(iosize=json_model['iosize'], learn_rate=json_model['learn_rate'])

        layers = json_model['layers']
        for layer in layers:
            model.add_layer(size=layer['size'],weights=layer['weights'],biases=layer['bias'], activation=layer['activation'])
    
        return model
    def clear_model(self):
        del self.layers

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
        json.dump(model, open(filename,'w'), sort_keys=True, indent=4)

    def predict(self, data, sess=None):
        if(sess==None):
            self.init_graph()
            with tf.Session() as sess:
                sess.run(self.init_op)
                return sess.run(self.output, feed_dict={self.x: data})
        else:
            return sess.run(self.output, feed_dict={self.x: data})

    # Returns a flattened version of the model in an array
    def flatten(self):
        m = []
        for layer in self.layers:
            weights = layer.get_weights()
            bias = layer.get_bias()
            print(weights.shape)
            print(bias.shape)

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

# Divisions = number of divisions of the database
def mnist_train(divisions=2, directory="models"):
    # Prep our output directory
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Division offsets of the database
    divs = int(60000/divisions)
    # Model settings
    epochs=100
    learn_rate = 0.0005
    # Make sure our models have the same starting weights/biases
    l1 = tf.Variable(tf.random_normal([784, 392], stddev=0.03))
    b1 = tf.Variable(tf.random_normal([392]))
    l2 = tf.Variable(tf.random_normal([392, 196], stddev=0.03))
    b2 = tf.Variable(tf.random_normal([196]))
    l3 = tf.Variable(tf.random_normal([196, 10], stddev=0.03))
    b3 = tf.Variable(tf.random_normal([10]))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        l1 = l1.eval()
        l2 = l2.eval()
        l3 = l3.eval()
        b1 = b1.eval()
        b2 = b2.eval()
        b3 = b3.eval()

    # Load the mnist dataset
    (x_dataset, y_labels), (x_test, test_labels) = load_mnist_data()
    # # x_dataset = x_dataset[:30000] # First 30k
    # # y_labels = y_labels[:30000]
    # x_dataset = x_dataset[30000:] # Second 30k
    # y_labels = y_labels[30000:]
    
    # Let's create the truth model.
    
    model = Model(iosize=[784,10])
    model.set_learn_rate(learn_rate)
    model.add_layer([784, 392], weights=l1, biases=b1)
    model.add_layer([392, 196], weights=l2, biases=b2)
    model.add_layer([196,10], activation='softmax', weights=l3, biases=b3)
    model.init_graph()

    model.train(x_dataset, y_labels, epochs=epochs, test_data=x_test, test_labels=test_labels, verbose=True)
    acc = model.accuracy_check(x_test, test_labels)
    model.model_accuracy = acc
    filename = "truth_model_" + str(divisions) + "_" + str(acc)[2:]
    model.save_model(filename)
    for i in range(divisions):

        # Create the model
        model = Model(iosize=[784,10])
        model.set_learn_rate(learn_rate)
        model.add_layer([784, 392], weights=l1, biases=b1)
        model.add_layer([392, 196], weights=l2, biases=b2)
        model.add_layer([196,10], activation='softmax', weights=l3, biases=b3)
        model.init_graph()

        # Create the dataset
        x_data = x_dataset[i*divs:(i+1)*divs]
        labels = y_labels[i*divs:(i+1)*divs]

        model.train(x_data, labels, epochs=epochs, test_data=x_test, test_labels=test_labels, verbose=True)

        print("Post training for model # " + str(i))
        acc = model.accuracy_check(x_test, test_labels)
        model.model_accuracy = acc
        filename = directory + "/mnist_model_" + str(i) + "_" + str(divisions) + "_" + str(acc)[2:]
        model.save_model(filename)

        # np.set_printoptions(suppress=True)
        # prediction = model.predict(x_test[:5])
        # print(prediction)
        # print("Labels")
        # print(test_labels[:5])
        # model.save_model(filename)
        # np.set_printoptions(suppress=True)

def recombine_unit_test():
    model = Model(iosize=[10,10])
    w = np.ones(10)
    w = w*2
    w1 = np.array([w,w,w,w,w,w,w,w,w,w])
    model.add_layer([10,10], weights=w1,biases=w)
    model.add_layer([10,10], weights=w1,biases=w)
    model.save_model("test2")

def recombine(file1, file2, fout):
    m1 = json.load(open(file1, "r"))
    m2 = json.load(open(file2, "r"))

    ml1 = m1['layers']
    ml2 = m2['layers']
    model = Model(iosize=m1['iosize'], learn_rate=m1['learn_rate'])
    layers = []
    for i in range(len(ml1)):
        layer = {}
        layer['activation'] = ml1[i]['activation']
        layer['size'] = ml1[i]['size']
        w1 = ml1[i]['weights']
        w2 = ml2[i]['weights']
        w1 = np.array(w1)
        w2 = np.array(w2)
        weights = w1+w2
        weights = weights*0.5
        weights = weights
        b1 = ml1[i]['bias']
        b2 = ml2[i]['bias']
        b1 = np.array(b1)
        b2 = np.array(b2)
        bias = b1 + b2
        bias = bias*0.5
        bias = bias
        layer['weights'] = weights
        layer['bias'] = bias
        model.add_layer(size=layer['size'], weights=layer['weights'],biases=layer['bias'], activation=layer['activation'])

    model.save_model(fout)
    return model

def test_mnist_model(filename):
    model = Model.load_from_file(filename)

    (x_dataset, y_labels), (x_test, test_labels) = load_mnist_data()

    print(x_test.shape)
    model.accuracy_check(x_test, test_labels)
    np.set_printoptions(suppress=True)
    prediction = model.predict(x_test[:5])
    print(prediction)
    print("Labels")
    print(test_labels[:5])

def bulk_naive_recombination(directory, file_out):
    print("Beginning naive recombination")
    # First thing is to get the name of all of the models we're going to be recombining.
    import glob
    files = []
    for file in glob.glob(directory + "/*"):
        files.append(file)
    num_files = len(files)
    print("Loading " + str(num_files) + " models.")

    # We'll store our temporary model info in a dictionary using the layer number as the key
    m_layers = {}

    iosize = [0,0]
    first_processed = False

    for file in files:
        f = json.load(open(file,"r"))
        layers = f['layers']

        if first_processed==False:
            iosize = f['iosize']
            learn_rate = f['learn_rate']
            first_processed=True

        for layer in layers:
            key = layer['_layer_num']
            if key in m_layers:
                m_layers[key]['weights'] = m_layers[key]['weights'] + np.array(layer['weights']) / num_files
                m_layers[key]['bias'] = m_layers[key]['bias'] + np.array(layer['bias']) / num_files
            else:
                # Create new entry if the layer doesn't exist yet
                m_layers[key] = {}
                m_layers[key]['weights'] = np.array(layer['weights']) / num_files
                m_layers[key]['bias'] = np.array(layer['bias']) / num_files

                m_layers[key]['size'] = layer['size']
                m_layers[key]['activation'] = layer['activation']


    model = Model(iosize=iosize, learn_rate=learn_rate)

    for i in m_layers.keys():
         model.add_layer(m_layers[i]['size'], weights=m_layers[i]['weights'], biases=m_layers[i]['bias'], activation=m_layers[i]['activation'])

    model.save_model(file_out)

def bulk_deviation_recombination(directory, file_out):
    print("Beginning deviation based recombination")
    # First thing is to get the name of all of the models we're going to be recombining.
    import glob
    files = []
    for file in glob.glob(directory + "/*"):
        files.append(file)
    num_files = len(files)
    print("Loading " + str(num_files) + " models.")

    # We'll store our temporary model info in a dictionary using the layer number as the key
    m_layers = {}

    iosize = [0,0]
    first_processed = False

    # Sucks, but I'm going to do this inefficiently since I didn't plan ahead for this
    avg_accuracy = 0.0
    highest_accuracy = 0.0
    for file in files:
        f = json.load(open(file,"r"))
        f_acc = float(f['accuracy'])
        avg_accuracy = avg_accuracy + f_acc / num_files
        if(highest_accuracy < f_acc):
            highest_accuracy = f_acc

    print("Average accuracy: " + str(avg_accuracy))
    print("Highest Accuracy: " + str(highest_accuracy))

    for file in files:
        f = json.load(open(file,"r"))
        layers = f['layers']

        if first_processed==False:
            iosize = f['iosize']
            learn_rate = f['learn_rate']
            first_processed=True
        f_acc = float(f['accuracy'])
        f_dev = 1 - abs(highest_accuracy - f_acc)
        # print("Deviation: " + str(f_dev))
        for layer in layers:
            key = layer['_layer_num']
            if key in m_layers:
                m_layers[key]['weights'] = m_layers[key]['weights'] + f_dev * np.array(layer['weights']) / num_files
                m_layers[key]['bias'] = m_layers[key]['bias'] + f_dev * np.array(layer['bias']) / num_files
            else:
                # Create new entry if the layer doesn't exist yet
                m_layers[key] = {}
                m_layers[key]['weights'] = f_dev * np.array(layer['weights']) / num_files
                m_layers[key]['bias'] = f_dev * np.array(layer['bias']) / num_files

                m_layers[key]['size'] = layer['size']
                m_layers[key]['activation'] = layer['activation']


    model = Model(iosize=iosize, learn_rate=learn_rate)

    for i in m_layers.keys():
         model.add_layer(m_layers[i]['size'], weights=m_layers[i]['weights'], biases=m_layers[i]['bias'], activation=m_layers[i]['activation'])

    model.save_model(file_out)


def recomb_nn():
    msize = 386718
    model = Model(iosize=[msize*2,msize])
    model.add_layer([msize*2, msize])
    model.add_layer([msize, msize])
    model.add_layer([msize, msize], activation='softmax')
    model.init_graph()

    model.flatten()

def flatten_test():
    model = Model.load_from_file("recombined_bulk_1000")
    model.init_graph()
    model.flatten()
m_divs = 1000
out_dir = "models/" + str(m_divs) + "div"
recombined_fname = "recombined_bulk_" + str(m_divs)
# Bulk create models
# mnist_train(divisions=m_divs, directory=out_dir)

# Bulk recombine those models
# bulk_naive_recombination(out_dir, recombined_fname)
# test_mnist_model("truth_model_1000_9199")
# bulk_deviation_recombination(out_dir, recombined_fname)
# Test the recombined model
# test_mnist_model(recombined_fname)

# recomb_nn()
flatten_test()
