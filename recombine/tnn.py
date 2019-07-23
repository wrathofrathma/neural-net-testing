#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from matplotlib import pyplot as plt
import json
import time
import os
import glob
from tensorflow.keras.datasets import mnist

from Model import Model

# Redoing custom neural network with tensorflow to avoid the input between layers from blowing up.
# Going to use eager execution for this for quicker prototyping. 
# tfe.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Loads mnist dataset into something useable by our neural network
def load_mnist_data():
    
    (x_dataset, y_labels), (x_test, y_test) = mnist.load_data()

    # print("Normalizing the dataset")
    x_dataset = x_dataset/255.0
    # print("Reshaping dataset")
    x_dataset = x_dataset.reshape(60000, 784)
    # print("Input dataset size: " + str(x_dataset.shape))
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
    # print("x_test size: " + str(x_test.shape))
    x_test = x_test.reshape(10000, 784)

    x_val = x_test[:5000]
    y_val = test_labels[:5000]
    x_test = x_test[5000:]
    test_labels = test_labels[5000:]
    return (x_dataset, y_labels), (x_test, test_labels), (x_val, y_val)

def bulk_naive_recombination(directory, file_out):
    # print("Beginning naive recombination")
    # First thing is to get the name of all of the models we're going to be recombining.
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

# Recombines a directory of neural networks into one other neural network of the same structure.
# Recombines based on an average of the weights multiplied by their accuracy in relation to the test set.
# Returns the accuracy in relation to the validation set. 

# def bulk_deviation_recombination(directory, file_out):
#     # print("Beginning deviation based recombination")
#     # First thing is to get the name of all of the models we're going to be recombining.
#     files = []
#     for file in glob.glob(directory + "/*"):
#         files.append(file)
#     num_files = len(files)
#     print("Loading " + str(num_files) + " models.")

#     # We'll store our temporary model info in a dictionary using the layer number as the key
#     m_layers = {}

#     iosize = [0,0]
#     first_processed = False

#     # Sucks, but I'm going to do this inefficiently since I didn't plan ahead for this
#     avg_accuracy = 0.0
#     highest_accuracy = 0.0
#     for file in files:
#         f = json.load(open(file,"r"))
#         f_acc = float(f['accuracy'])
#         avg_accuracy = avg_accuracy + f_acc / num_files
#         if(highest_accuracy < f_acc):
#             highest_accuracy = f_acc

#     print("Average accuracy: " + str(avg_accuracy))
#     print("Highest Accuracy: " + str(highest_accuracy))

#     for file in files:
#         f = json.load(open(file,"r"))
#         layers = f['layers']

#         if first_processed==False:
#             iosize = f['iosize']
#             learn_rate = f['learn_rate']
#             first_processed=True
#         f_acc = float(f['accuracy'])
#         f_dev = 1 - abs(highest_accuracy - f_acc)
#         # print("Deviation: " + str(f_dev))
#         for layer in layers:
#             key = layer['_layer_num']
#             if key in m_layers:
#                 m_layers[key]['weights'] = m_layers[key]['weights'] + f_dev * np.array(layer['weights']) / num_files
#                 m_layers[key]['bias'] = m_layers[key]['bias'] + f_dev * np.array(layer['bias']) / num_files
#             else:
#                 # Create new entry if the layer doesn't exist yet
#                 m_layers[key] = {}
#                 m_layers[key]['weights'] = f_dev * np.array(layer['weights']) / num_files
#                 m_layers[key]['bias'] = f_dev * np.array(layer['bias']) / num_files

#                 m_layers[key]['size'] = layer['size']
#                 m_layers[key]['activation'] = layer['activation']


#     model = Model(iosize=iosize, learn_rate=learn_rate)

#     for i in m_layers.keys():
#          model.add_layer(m_layers[i]['size'], weights=m_layers[i]['weights'], biases=m_layers[i]['bias'], activation=m_layers[i]['activation'])

#     model.save_model(file_out)


# Bulk trains models passed in a python list
# returns time spent training each model
def bulk_train(models, epochs):

    num_models = len(models)
    divs = int(60000/num_models)
    (x_dataset, y_labels), (x_test, test_labels), (x_val, y_val) = load_mnist_data()

    times = []
    for i in range(num_models):
        model = models[i]
        s = time.time()
        model.init_graph()
        x_data = x_dataset[i*divs:(i+1)*divs]
        labels = y_labels[i*divs:(i+1)*divs]
        model.train(x_data, labels, epochs=epochs, test_data=x_test, test_labels=test_labels, verbose=False)
        f = time.time()
        times.append(f-s)
        model.save_model()
    return times
# Bulk assess accuracy. Returns an array of accuracies of each model based on the validation set
def bulk_accuracy_check(models, data, labels, store=False):
    accuracy = []
    for model in models:
        accuracy.append(model.accuracy_check(data, labels, store))
    return accuracy

# Bulk opens models so we don't have to create them every time.
def bulk_open_models(directory):
    models = []
    if not os.path.exists(directory):
        print("Directory doesn't exist")
        exit(1)

    files = []
    for file in glob.glob(directory + "/*"):
        files.append(file)

    for file in files:
        models.append(Model.load_from_file(file))
    return models

# Creates a bunch of bulk models based on the same weights and divisions.
# Outputs a directory with a truth model, and a subdirectory of smaller models.
# Returns the time it took to train the truth model 100 epochs
def create_bulk_models(directory="model", divisions=2):
    # Prep our output directories
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory + "/models"):
        os.makedirs(directory + "/models")

    # Division offsets of the database
    divs = int(60000/divisions)
    # Model settings
    epochs=100
    learn_rate = 0.0005
    # Initialize the layer weights & biases to be the same for all models.
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

    # Load the mnist dataset for training the first one.
    (x_dataset, y_labels), (x_test, test_labels), (x_val, y_val) = load_mnist_data()
    # Let's create the single model.
    filename = directory + "/full_model"
    model = Model(filename=filename, iosize=[784,10])
    model.set_learn_rate(learn_rate)
    model.add_layer([784, 392], weights=l1, biases=b1)
    model.add_layer([392, 196], weights=l2, biases=b2)
    model.add_layer([196,10], activation='softmax', weights=l3, biases=b3)
    model.init_graph()

    # model.train(x_dataset, y_labels, epochs=epochs, test_data=x_test, test_labels=test_labels, verbose=False)
    with tf.Session() as sess:
        sess.run(model.init_op)
        model.store_weights_biases(sess)
    model.save_model(filename)

    models = []
    # Bulk create the models 
    for i in range(divisions):

        # Create the model
        filename = directory + "/models/mnist_model_" + str(i) + "_" + str(divisions)
        models.append(Model(filename=filename, iosize=[784,10]))
        models[-1].set_learn_rate(learn_rate)
        # models[-1].add_layer([784, 392])
        # models[-1].add_layer([392, 196])
        # models[-1].add_layer([196,10], activation='softmax')
        models[-1].add_layer([784, 392], weights=l1, biases=b1)
        models[-1].add_layer([392, 196], weights=l2, biases=b2)
        models[-1].add_layer([196,10], activation='softmax', weights=l3, biases=b3)
        models[-1].init_graph()

        with tf.Session() as sess:
            sess.run(models[-1].init_op)
            models[-1].store_weights_biases(sess)

        # Create the dataset
        # x_data = x_dataset[i*divs:(i+1)*divs]
        # labels = y_labels[i*divs:(i+1)*divs]

        # model.train(x_data, labels, epochs=epochs test_data=x_test, test_labels=test_labels, verbose=True)

        # print("Post training for model # " + str(i))
        acc = models[-1].accuracy_check(x_test, test_labels, True)
        models[-1].save_model(filename)
    models.append(model)
    return models

# Automates the bulk training and generates/collects the time/accuracy data over training epochs
def auto_bulk_train(directory, divisions):
    model_sub_dir = directory + "/models"
    # model file names
    recomb_fname = directory + "/recomb_model"
    full_fname = directory + "/full_model"

    epoch_intervals = 5
    # Creates/initialies the models with the same weights
    models = create_bulk_models(directory,divisions)
    full_model = models[-1] # Extract the full/truth model
    # models = models[:-1] # remove it from the list so when we recombine stuff later it's not included

    # initial accuracy states
    (x_dataset, y_labels), (x_test, test_labels), (x_val, y_val) = load_mnist_data()

    accuracy = bulk_accuracy_check(models, x_val, y_val, False)


    full_accuracy = full_model.accuracy_check(x_val, y_val, False)

    bulk_naive_recombination(model_sub_dir, recomb_fname)

    recomb_model = Model.load_from_file(recomb_fname)

    recomb_accuracy = recomb_model.accuracy_check(x_val, y_val)

    print("Full model's accuracy on validation set: " + str(full_accuracy))
    print("Recomb accuracy: " + str(recomb_accuracy))

    epoch_count = 0
    accuracy_tracker = []
    accuracy_tracker.append(accuracy)
    recomb_tracker = []
    recomb_tracker.append(recomb_accuracy)
    while(recomb_accuracy < full_accuracy or (recomb_accuracy < 0.92 and full_accuracy < 0.92)):
        bulk_train(models[:-1], epoch_intervals)
        full_model.init_graph()
        full_model.train(x_dataset, y_labels, epochs=epoch_intervals, test_data=x_test, test_labels=test_labels, verbose=False)
        accuracy = bulk_accuracy_check(models, x_val, y_val, False)
        accuracy_tracker.append(accuracy)
        
        bulk_naive_recombination(model_sub_dir, recomb_fname)

        recomb_model = Model.load_from_file(recomb_fname)
        recomb_accuracy = recomb_model.accuracy_check(x_val, y_val)
        recomb_tracker.append(recomb_accuracy)

        print("##################### BATCH REPORT #####################")
        epoch_count+=5
        print("Total epochs: " + str(epoch_count))
        print("Accuracies of individual models on validation set")
        print(accuracy[:-1])
        print("Full accuracy: " + str(accuracy[-1]))
        print("Recomb accuracy: " + str(recomb_accuracy))
        if(epoch_count>=100):
            break

    print("##################### Training REPORT #####################")
    print("Total epochs: " + str(epoch_count))
    print("Accuracies of individual models on validation set")
    print(accuracy[:-1])
    print("Full model accuracy: " + str(accuracy[-1]))
    print("Recomb accuracy: " + str(recomb_accuracy))
    print("Accuracy_tracker")
    for line in accuracy_tracker:
        print(*line, sep=' ')
    print("Recomb Tracker")
    for line in recomb_tracker:
        print(*recomb_tracker, sep=' ')

def bulk_open_models(directory):
    models = []
    if not os.path.exists(directory):
        print("Directory doesn't exist")
        exit(1)

    files = []
    for file in glob.glob(directory + "/*"):
        files.append(file)

    for file in files:
        models.append(Model.load_from_file(file))
    return models

model_dir = "models/32_split"
auto_bulk_train(model_dir,32)

