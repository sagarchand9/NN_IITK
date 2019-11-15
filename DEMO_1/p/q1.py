#################################
#Author: Pawan Harendra Mishra
#EE671A: Neural Networks
#Question 1: Backpropagation with
#momentum
#################################

from numpy import *
from matplotlib.pyplot import *

#training data from csv file
data = genfromtxt("data1.csv", delimiter = ',')

#global parameters
lr = 0.001 #learning rate
a = 0.0005 #momentum constant
epsilon = 3 #convergence
n, m = shape(data)
l0  = m - 2 #number of input layer nodes
l1 = 5 #number of first layer nodes
l2 = 2 #nuber of second layer nodes
#l3 = # #number of third layer nodes
#l4 = # #number of fourth layer nodes

#sigmoidal activation function
def sigmoid(x):
    return 1/(1 + exp(-x))

#function for training the network
def training():
    x0 = data[:,2:m] #input data
    yd = data[:,:2] #output data

    #random initialization of weights between -1 and 1
    w0 = 2*random.random((l0, l1)) - 1
    w0Prev = zeros((l0, l1))
    w1 = 2*random.random((l1, l2)) - 1
    w1Prev = zeros((l1, l2))

    it = 0
    error = []

    while True:
        #forward propagation
        it += 1
        h0 = dot(x0, w0)
        x1 = sigmoid(h0)
        h1 = dot(x1, w1)
        x2 = sigmoid(h1)
        e = yd - x2

        #back propagation
        d1 = e*x2*(1-x2)
        w1Curr = w1
        w1 = w1 + lr*dot(x1.T, d1) + a*(w1 - w1Prev)
        w1Prev = w1Curr

        d0 = x1*(1-x1)*dot(d1, w1.T)
        w0Curr = w0
        w0 = w0 + lr*dot(x0.T, d0) + a*(w0 - w0Prev)
        w0Prev = w0Curr

        error.append(sum(e**2))

        if it%50 == 0:
            print it, error[-1]

        if error[-1] < epsilon or it > 100000:
            break

    print error[-1]
    return w0, w1, error


w0, w1, error = training()
#save trained weights
savetxt("q1_w0.csv", w0, delimiter = ',')
savetxt("q1_w1.csv", w1, delimiter = ',')

yd = data[:,:2] #output data
x0 = data[:,2:m] #input data
h0 = dot(x0, w0)
x1 = sigmoid(h0)
h1 = dot(x1, w1)
x2 = sigmoid(h1)

#plotting
f1 = figure()
ax1 = f1.add_subplot(1,1,1)
ax1.plot(array(range(1, n+1)), yd[:, 0], color = 'r', label = "Vx Data")
ax1.plot(array(range(1, n+1)), x2[:, 0], color = 'b', label = "Vx Output")
xlabel("Data Point")
ylabel("Vx")
ax1.legend()
title("Question 1: Vx")
savefig("q1_1.png")

f2 = figure()
ax2 = f2.add_subplot(1,1,1)
ax2.plot(array(range(1, n+1)), yd[:, 1], color = 'r', label = "Vy Data")
ax2.plot(array(range(1, n+1)), x2[:, 1], color = 'b', label = "Vy Output")
xlabel("Data Point")
ylabel("Vy")
ax2.legend()
title("Question 1: Vy")
savefig("q1_2.png")

n = len(error)
f3 = figure()
ax3 = f3.add_subplot(1,1,1)
ax3.plot(array(range(1, n+1)), array(error), color = 'r')
xlabel("iterations")
ylabel("error")
title("Question 1: Error Vs. Iterations")
savefig("q1_3.png")
