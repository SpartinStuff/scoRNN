# scoRNN

Code for the paper "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform", https://arxiv.org/abs/1707.09520. If you find this code useful, please cite the paper.

Uses Tensorflow. To run, download the desired experiment code as well as the "scoRNN.py" script. 

Each script uses command line arguments to specify the desired architecture. For example, to run the MNIST experiment with a hidden size of 170 and RMSprop optimizer with learning rate 1e-3 to update the recurrent weight, type in the command line: 

python scoRNN_MNIST.py 170 rmsprop 1e-3
