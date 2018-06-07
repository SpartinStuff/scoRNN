'''
A Recurrent Neural Network (RNN) implementation using TensorFlow library.  Can
be used to run the scoRNN architecture and an LSTM on the adding problem.  The
adding problem consists of two concurrent sequences.  The first sequence is a 
random sequence of numbers and the other is a marker of zeros and ones.  The 
digits marked with one are to be added together.
'''


# Import modules
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import sys

from scoRNN import *


# Network Parameters
model = 'scoRNN'
n_input = 2             
n_steps = 200           # Length of sequence
n_hidden = 170          # Hidden layer size
n_neg_ones = 85         # No. of -1's to put on diagonal of scaling matrix
n_classes = 1           # One output (sum of two numbers)
training_epochs = 10
batch_size = 50
training_size = 100000   # Training set size
testing_size = 10000     # Testing set size
display_step = 100


# Input/Output parameters
in_out_optimizer = 'adam'
in_out_lr = 1e-3


# Hidden to hidden parameters
A_optimizer = 'rmsprop'
A_lr = 1e-4


# COMMAND LINE ARGS: MODEL HIDDENSIZE IO-OPT IO-LR AOPT ALR NEG-ONES SQLENGTH
try:
    model = sys.argv[1]
    n_hidden = int(sys.argv[2])
    in_out_optimizer = sys.argv[3]
    in_out_lr = float(sys.argv[4])
    A_optimizer = sys.argv[5]
    A_lr = float(sys.argv[6])
    n_neg_ones = int(sys.argv[7])
    n_steps = int(sys.argv[8])
except IndexError:
    pass


# Setting the random seed
tf.set_random_seed(5544)
np.random.seed(5544)


# Name of save string/scaling matrix
if model == 'LSTM':
    savestring = '{:s}_{:d}_{:s}_{:.1e}_seq_length_{:d}'.format(model, n_hidden, \
                 in_out_optimizer, in_out_lr, n_steps)
if model == 'scoRNN':
    savestring = '{:s}_{:d}_{:d}_{:s}_{:.1e}_{:s}_{:.1e}_seq_length_{:d}'.format(model, \
                 n_hidden, n_neg_ones, in_out_optimizer, in_out_lr, \
                 A_optimizer, A_lr, n_steps)
    D = np.diag(np.concatenate([np.ones(n_hidden - n_neg_ones), \
        -np.ones(n_neg_ones)]))
print('\n')
print(savestring)
print('\n')


# Generates Synthetic Data
def Generate_Data(size, length):
    
    # Random sequence of numbers
    x_random = np.random.uniform(0,1, size = [size, length])

    # Random sequence of zeros and ones
    x_placeholders = np.zeros((size, length))
    firsthalf = int(np.floor((length-1)/2.0))
    for i in range(0,size):
        x_placeholders[i, np.random.randint(0, firsthalf)] = 1
        x_placeholders[i, np.random.randint(firsthalf, length)] = 1

    # Create labels
    y_labels = np.reshape(np.sum(x_random*x_placeholders, axis=1), (size,1))
    
    # Creating data with dimensions (batch size, n_steps, n_input)
    data = np.dstack((x_random, x_placeholders))
    
    return data, y_labels


# Defining RNN architecture
def RNN(x):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
   
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)
    
    
    # Create RNN cell
    if model == 'LSTM':
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, activation=tf.nn.tanh)
    
    if model == 'scoRNN':
        rnn_cell = scoRNNCell(n_hidden, D = D)
    
    # Place RNN cell into RNN, take last timestep as output    
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    rnnoutput = outputs[-1]
    
        
    # Last layer, linear
    output = tf.layers.dense(inputs=rnnoutput, units=n_classes, activation=None)
    return output


# Used to calculate Cayley Transform derivative
def Cayley_Transform_Deriv(grads, A, W):
    

    # Calculate Update Matrix
    I = np.identity(grads.shape[0])
    Temporary = np.linalg.lstsq(I+A,I)[0]
    Update = np.dot(np.dot(Temporary.T, grads), D + W.T) 
    DFA = Update.T - Update    

  

    return DFA


# Used to make the hidden to hidden weight matrix
def makeW(A):
    # Computing hidden to hidden matrix using the relation 
    # W = (I + A)^-1*(I – A)D
    
    I = np.identity(A.shape[0])
    W = np.dot(np.linalg.lstsq(I + A, I - A)[0], D)
    
    return W


# Plotting MSE
def graphlosses(tr_mse, te_mse, xax):
    
    xax = [x/100000.0 for x in xax]
    
    plt.plot(xax, tr_mse, label='training MSE')
    plt.plot(xax, te_mse, label='testing MSE')
    plt.ylim([0,0.25])
    plt.legend(loc='lower left', prop={'size':6})
    plt.savefig(savestring + '.png')
    
    plt.clf()
    
    return


# Generating training and test data
x_train, y_train = Generate_Data(training_size, n_steps)
test_data, test_label = Generate_Data(testing_size, n_steps)


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


# Assigningto RNN function
pred = RNN(x)

        
# Define loss object
cost = tf.reduce_mean(tf.squared_difference(pred, y))


# Optimizers/Gradients
optimizer_dict = {'adam' : tf.train.AdamOptimizer,
                  'adagrad' : tf.train.AdagradOptimizer,
                  'rmsprop' : tf.train.RMSPropOptimizer,
                  'sgd' : tf.train.GradientDescentOptimizer}
        
opt1 = optimizer_dict[in_out_optimizer](learning_rate=in_out_lr)        


# LSTM training operations
if model == 'LSTM':
    LSTMtrain = opt1.minimize(cost)


# scoRNN training operations
if model == 'scoRNN':
    opt2 = optimizer_dict[A_optimizer](learning_rate=A_lr)
    Wvar = [v for v in tf.trainable_variables() if 'W:0' in v.name][0]
    Avar = [v for v in tf.trainable_variables() if 'A:0' in v.name][0]
    othervarlist = [v for v in tf.trainable_variables() if v not in [Wvar, Avar]]

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


# Plotting lists
number_iters_plt = []
train_mse_plt = []
test_mse_plt = []


# Training
with tf.Session() as sess:  


    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)


    # Get initial A and W    
    if model == 'scoRNN':    
        A, W = sess.run([Avar, Wvar])
        

    # Keep training until reach number of epochs
    epoch = 1
    while epoch <= training_epochs:
        step = 1
        # Keep training until reach max iterations
        while step * batch_size <= training_size:
                                  
            
            # Getting input data
            batch_x = x_train[(step-1)*batch_size:step*batch_size,:,:]
            batch_y = y_train[(step-1)*batch_size:step*batch_size]   
            
                   
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

                        
            # Evaluating the MSE of the model
            if step % display_step == 0:
                
                # Evaluating train and test MSE.   
                train_mse = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                test_mse = sess.run(cost, feed_dict={x: test_data, y: test_label})
                

                # Printing results
                print('\n')
                print("Epoch:", epoch)
                print("Percent complete:", step*batch_size/training_size) 
                print("Training Minibatch MSE:", train_mse)
                print("Test MSE:", test_mse)
                    

                # Plotting
                number_iters_plt.append(step*batch_size + \
 		                           (epoch-1)*training_size)
                train_mse_plt.append(train_mse)
                test_mse_plt.append(test_mse)
                    
                graphlosses(train_mse_plt, test_mse_plt, number_iters_plt)

                np.savetxt(savestring + '_train_MSE.csv', train_mse_plt, \
                		  delimiter=',')
                np.savetxt(savestring + '_test_MSE.csv', test_mse_plt, \
		                 delimiter = ',')
                np.savetxt(savestring + '_iters.csv', number_iters_plt, \
		                  delimiter = ',')
             
            step += 1                   
        epoch += 1
            
            
            
    print("Optimization Finished!")
        
