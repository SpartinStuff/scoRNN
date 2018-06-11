'''
A Recurrent Neural Network (RNN) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits 
(http://yann.lecun.com/exdb/mnist/)
'''

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sps

import sys
import os
import time

from scoRNN import *

'''
Input is a sequence of digits to copy, followed by a string of T 0s, 
a marker (we use 9) to begin printing the copied sequence, and more zeroes.
Target output is all zeroes until the marker, at which point the machine
should print the sequence it saw at the beginning.

Example for T = 10 and n_sequence = 5:

Input
3 6 5 7 2 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0
Target output
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 6 5 7 2
'''


# Network Parameters
model = 'scoRNN'
T = 1000          # number of zeroes to put between sequence and marker
n_sequence = 10  # length of sequence to copy
n_input = 10          # number of possible inputs (0-9)
n_classes = 9          # number of possible output classes (0-8, 9 is not a valid output)
n_hidden = 190          # hidden layer num of features
n_neg_ones = 95          # number of negative eigenvalues to put on diagonal
train_size = 20000
test_size = 1000
batch_size = 20
display_step = 50
iterations = 10000

# Input/Output Parameters
in_out_optimizer = 'rmsprop'
in_out_lr = 1e-3

# Hidden to Hidden Parameters
A_optimizer = 'rmsprop'
A_lr = 1e-4



# COMMAND LINE ARGS: MODEL TFOPT AOPT TFLR ALR HIDDENSIZE NEGEIGS MRELU
try:
    model = sys.argv[1]
    n_hidden = int(sys.argv[2])
    in_out_optimizer = sys.argv[3]
    in_out_lr = float(sys.argv[4])
    A_optimizer = sys.argv[5]
    A_lr = float(sys.argv[6])
    n_neg_ones = int(sys.argv[7])
    T = int(sys.argv[8])
except IndexError:
    pass

n_steps = n_sequence*2 + T

baseline = (n_sequence*np.log(n_classes-1))/(T + 2*n_sequence)

A = np.zeros((n_hidden, n_hidden))

# Setting the random seed
tf.set_random_seed(5544)
np.random.seed(5544)


# Plotting Commands
number_iters_plt = []
train_loss_plt = []
test_loss_plt = []
train_accuracy_plt = []
test_accuracy_plt = []


# name of graph file
if model == 'LSTM':
    savestring = '{:s}_{:d}_{:s}_{:.1e}_T={:d}'.format(model, n_hidden, \
                 in_out_optimizer, in_out_lr, T)
if model == 'scoRNN':
    savestring = '{:s}_{:d}_{:d}_{:s}_{:.1e}_{:s}_{:.1e}_T={:d}'.format(model, \
                 n_hidden, n_neg_ones, in_out_optimizer, in_out_lr, \
                 A_optimizer, A_lr, T)
    D = np.diag(np.concatenate([np.ones(n_hidden - n_neg_ones), \
        -np.ones(n_neg_ones)]))

print('\n' + savestring + '\n')


def copying_data(T, seq):
    n_data, n_sequence = seq.shape
    
    zeros1 = np.zeros((n_data, T-1))
    zeros2 = np.zeros((n_data, T))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))
    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int64')
    
    return x, y


# Defining RNN architecture
def RNN(x):

    # Create RNN cell
    if model == 'LSTM':
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, activation=tf.nn.tanh)
    
    if model == 'scoRNN':
        rnn_cell = scoRNNCell(n_hidden, D = D)
    
    # Place RNN cell into RNN, take last timestep as output
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    
    with tf.variable_scope("output"):
        weights = tf.get_variable("weights", shape=[n_hidden, n_classes])
        biases = tf.get_variable("bias", shape=[n_classes])
    
    temp_out = tf.map_fn(lambda r: tf.matmul(r, weights), outputs)
    output_data = tf.nn.bias_add(temp_out, biases)
    return output_data
    
    
    # last layer, linear
    output = tf.layers.dense(inputs=rnnoutput, units=n_classes, activation=None)
    return output

# Used to calculate Cayley Transform derivative
def Cayley_Transform_Deriv(grads, A, W):
    
    # Calculate update matrix
    I = np.identity(grads.shape[0])
    Update = np.linalg.lstsq((I + A).T, np.dot(grads, D + W.T))[0]
    DFA = Update.T - Update
    
    return DFA


# Used to make the hidden to hidden weight matrix
def makeW(A):
    # Computing hidden to hidden matrix using the relation 
    # W = (I + A)^-1*(I - A)D
    
    I = np.identity(A.shape[0])
    W = np.dot(np.linalg.lstsq(I+A, I-A)[0],D)  

    return W

# Used for printing values
def getprintvalues(A, W):
    I = np.identity(A.shape[0])
    orthogonalcheck = np.linalg.norm(I - np.dot(W.T,W))
    A_norm = np.linalg.norm(A, ord=1)
    IA_inverse = np.linalg.lstsq(I + A, I)
    IA_inverse_norm = np.linalg.norm(IA_inverse[0], ord=1)
    IW_norm = np.linalg.norm(I + W, ord=1)
    
    return orthogonalcheck, A_norm, IA_inverse_norm, IW_norm

def graphlosses(tr_loss, te_loss, tr_acc, te_acc, xax):
    
    plt.plot(xax, tr_loss, label='training loss')
    plt.plot(xax, te_loss, label='testing loss')
    plt.plot((xax[0], xax[-1]), (baseline, baseline), linestyle='-')
    plt.ylim([0,baseline*2])
    plt.legend(loc='lower left', prop={'size':6})
    #plt.subplot(2,1,2)
    #plt.plot(xax, tr_acc, label='training acc')
    #plt.plot(xax, te_acc, label='testing acc')
    #plt.legend(loc='upper left', prop={'size':6})
    #plt.ylim([0.9,1])
    plt.savefig(savestring + ".png")
    plt.clf()
    
    return


# Graph input
x = tf.placeholder("int32", [None, n_steps])
y = tf.placeholder("int64", [None, n_steps])
    
inputdata = tf.one_hot(x, n_input, dtype=tf.float32)

#inputdata = tf.one_hot(x, n_input, dtype=tf.float32)

# Assigning variable to RNN function
pred = RNN(inputdata)

# Cost & accuracy
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))



correct_pred = tf.equal(tf.argmax(pred, 2), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Define optimizer object
optimizer_dict = {'adam' : tf.train.AdamOptimizer,
                'adagrad' : tf.train.AdagradOptimizer,
                'rmsprop' : tf.train.RMSPropOptimizer,
                'sgd' : tf.train.GradientDescentOptimizer}

opt1 = optimizer_dict[in_out_optimizer](learning_rate=in_out_lr)

if model == 'LSTM':
    LSTMtrain = opt1.minimize(cost)

if model == 'scoRNN':
    opt2 = optimizer_dict[A_optimizer](learning_rate=A_lr)
    Wvar = [v for v in tf.trainable_variables() if 'W:0' in v.name][0]
    Avar = [v for v in tf.trainable_variables() if 'A:0' in v.name][0]
    othervarlist = [v for v in tf.trainable_variables() if v not in \
                [Wvar, Avar]]
    
    # Getting gradients
    grads = tf.gradients(cost, othervarlist + [Wvar])

    # Applying gradients to input-output weights
    with tf.control_dependencies(grads):
        applygrad1 = opt1.apply_gradients(zip(grads[:len(othervarlist)], \
                    othervarlist))  
    
    # Updating variables
    newW = tf.placeholder(tf.float32, Wvar.get_shape())
    updateW = tf.assign(Wvar, newW)
    
    # Applying hidden-to-hidden gradients
    gradA = tf.placeholder(tf.float32, Avar.get_shape())
    applygradA = opt2.apply_gradients([(gradA, Avar)])
    


with tf.Session() as sess:
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Get initial A and W
    if model == 'scoRNN':
        A, W = sess.run([Avar, Wvar])
    
    # Launch the graph, tensorboard log, saver
    #datestring = time.strftime("%Y%m%d", time.gmtime())
    #if not os.path.exists("./savedmodels/" + datestring):
        #os.makedirs("./savedmodels/" + datestring)
    #sess.run(init)
    #saver = tf.train.Saver()
    #writer = tf.summary.FileWriter('./graphs', sess.graph)
    
    
    
    training_seq = np.random.randint(1, high=9, size=(train_size, n_sequence))
    test_seq = np.random.randint(1, high=9, size=(test_size, n_sequence))
    test_seq = np.split(test_seq, 10)
    #test_x, test_y = copying_data(T, test_seq)
    
    # Keep training until reach number of iterations
    step = 0
    while step < iterations:
        # input data
        batch_seq = training_seq[(step*batch_size) % train_size:((step*batch_size) % train_size) + batch_size]
        batch_x, batch_y = copying_data(T, batch_seq)
        
        # Updating weights
        if model == 'LSTM':
            sess.run(LSTMtrain, feed_dict={x: batch_x, y: batch_y})


        if model == 'scoRNN':
            _, hidden_grads = sess.run([applygrad1, grads[-1]], \
                                feed_dict = {x: batch_x, y: batch_y})
            DFA = Cayley_Transform_Deriv(hidden_grads, A, W)            
            sess.run(applygradA, feed_dict = {gradA: DFA})
            A = sess.run(Avar)
            W = makeW(A)
            sess.run(updateW, feed_dict = {newW: W})
            
        step += 1
        
        if step % display_step == 0:
                    
            # Printing commands
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
            
            test_metrics = []
            for seq in test_seq:
                tseq_x, tseq_y = copying_data(T, seq)
                test_metrics.append(sess.run([accuracy, cost], feed_dict={x: tseq_x, y: tseq_y}))
            test_acc, test_loss = np.mean(test_metrics, axis=0)
            
            print('\n')
            print("Iterations: ", step)
            print("Testing Accuracy:", test_acc)
            print("Testing Loss:", test_loss)
            print("Training Accuracy:", acc)
            print("Training Loss:", loss)
            print("Baseline:", baseline)
            print('\n')
            
            # Plotting
            number_iters_plt.append(step)
            train_loss_plt.append(loss)
            test_loss_plt.append(test_loss)
            train_accuracy_plt.append(acc)
            test_accuracy_plt.append(test_acc)
            
            graphlosses(train_loss_plt, test_loss_plt, train_accuracy_plt, test_accuracy_plt, number_iters_plt)
            
print("Optimization Finished!")
        
