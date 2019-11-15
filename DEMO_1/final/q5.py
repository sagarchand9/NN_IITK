from numpy import *
from matplotlib.pyplot import *
import os

#training data from csv file
trainData = genfromtxt("data1.csv", delimiter = ',')
#testing data from csv file
testData = genfromtxt("data3.csv", delimiter = ',')

#global parameters
l, m = shape(trainData)
lr0 = 0.005 #learning rate parameter for weight
#yLambda and aLambda are initialized below
lr1 = 0.01 #learning rate parameter for yLambda
lr2 = 0.01 #learning rate parameter for aLambda
t1 = 10 #time constant of learning rate
t2 = 10 #time constant of width function
sig0 = 1 #initial value of width
sz = 25 #size of 1-D SOM
maxIt = 200 #maximum training steps
epsilon = 0.3 #Convergence

x0 = trainData[:,2:m] #clustering data
yd = trainData[:,:2] #output data

d = empty(sz) #weight-input distance array
nd = empty((sz,1)) #neuron distance array

#function to compute euclidean distance
def dist(i, j):
    return sqrt(sum((i - j)**2))

#learning rate for n-th training step
def lr(n):
    return lr0*exp(-n/t2)

#width for n-th training step
def sig(n):
    return sig0*exp(-n/t1)

#neighbourhood function for n-th training step
def gaussian(n, d):
    return exp(-d**2/(2*sig(n)**2))

#function for unsupervised mapping
def clustering():
    #random initialization of weights between -1 and 1
    w = 2*random.random((sz, m - 2)) - 1
    #values associated with each neuron
    yLambda = random.random((sz, 2))
    aLambda = random.random((2, m - 2))

    #plot initialized weights
    f1 = figure()
    ax1 = f1.add_subplot(1,1,1)
    title("Weights before training (Vx, Vy only; not s_3, s_4 and s_5)")
    ax1.scatter(w[:,3], w[:,4])
    savefig("q5_1.png")

    n = 0 #training steps
    error = zeros(maxIt)

    while True:
        n += 1
        for j in range(l):
            #loop to compute distance vector for each neuron
            for i in range(sz):
                d[i] = dist(w[i], x0[j])

            #index of winning neuron
            i = argmin(d)

            for k in range(sz):
                #distance from winning neuron
                nd[k] = abs(k - i)

            #learing rate for
            neta = lr(n)
            h = gaussian(n, nd)

            #collective response
            s = sum(h)
            y = sum(h*(yLambda + dot((x0[j] - w), aLambda.T)/s), axis=0)

            e = yd[j] - y
            error[n-1] += sum(e**2)

            for k in range(sz):
                #weight update
                w[k] = w[k] + neta*h[k]*(x0[j] - w[k])

            #yLambda and aLambda update
            yLambda = yLambda + lr1*e*h/s
            aLambda = aLambda + lr2*dot((e*h).T, (x0[j] - w))/s

        error[n-1] = sqrt(error[n-1])
        print n, error[n-1]
        if error[n-1] < (epsilon) or n >= maxIt:
            break

    #analyse clusters
    neuronClustering = zeros((sz,1), dtype=int)
    clusterSum = zeros((sz, 2))

    for j in range(l):
        #compute distance vector for each neuron
        for i in range(sz):
            d[i] = dist(w[i], x0[j])

        #index of winning neuron
        i = argmin(d)

        #segregation
        neuronClustering[i] += 1
        clusterSum[i] += yd[j]

    clusterCount = 0
    clusterMean = zeros((sz, 2))

    #compute mean Vx and Vy for each cluster formed
    for i in range(sz):
        if neuronClustering[i] != 0:
            clusterMean[i] = clusterSum[i] / neuronClustering[i]
            clusterCount += 1

    clusterSizes = zeros(clusterCount, dtype=int)
    clusterLabels = zeros((clusterCount, 2))

    it = 0
    for i in range(sz):
        if neuronClustering[i] != 0:
            clusterSizes[it] = neuronClustering[i]
            clusterLabels[it] = clusterMean[i]
            it += 1

    #Write cluster data to file
    with open("q5_meta.txt", "w") as f:
        f.write("Number of Neurons: {}\n\n".format(sz))
        f.write("Number of Clusters formed: {}\n\n".format(clusterCount))
        f.write("Size of each Cluster:\n{}\n\n".format(clusterSizes))
        f.write("Labels for each cluster (Vx, Vy):\n{}\n\n".format(clusterLabels))

    return w, yLambda, aLambda, error

n = 1

flag = True
if os.path.isfile("q5_w.csv") and os.path.isfile("q5_yLambda.csv") and os.path.isfile("q5_aLambda.csv"):
    print "Pretrained weights and neuron values exist. Do you want to use them?"
    response = raw_input("[Y/N?]: ")
    if response == "Y" or response == "y":
        w = np.genfromtxt("q5_w.csv", delimiter=",")
        yLambda = np.genfromtxt ("q5_yLambda.csv", delimiter=",")
        aLambda = np.genfromtxt ("q5_aLambda.csv", delimiter=",")
        flag = False

if flag:
    w, yLambda, aLambda, error = clustering()
    savetxt("q5_w.csv", w, delimiter = ',')
    savetxt("q5_yLambda.csv", yLambda, delimiter = ',')
    savetxt("q5_aLambda.csv", aLambda, delimiter = ',')

    y = empty((l,2)) #output array

    for j in range(l):
        #loop to compute distance vector for each neuron
        for i in range(sz):
            d[i] = dist(w[i], x0[j])

        #index of winning neuron
        i = argmin(d)

        for k in range(sz):
            #distance from winning neuron
            nd[k] = abs(k - i)

        h = gaussian(n, nd)
        s = sum(h)

        #compute output for each input data
        y[j] = sum(h*(yLambda + dot((x0[j] - w), aLambda.T)/s), axis=0)

    #plot output and actual data
    f2 = figure()
    ax2 = f2.add_subplot(1,1,1)
    title("Question 5: Vx trained")
    xlabel("Data Point")
    ylabel("Vx")
    ax2.plot(array(range(1, l+1)), yd[:, 0], color = 'r', label = "Vx Data")
    ax2.plot(array(range(1, l+1)), y[:, 0], color = 'b', label = "Vx trained output")
    ax2.legend()
    savefig("q5_2.png")

    f3 = figure()
    ax3 = f3.add_subplot(1,1,1)
    title("Question 5: Vy trained")
    xlabel("Data Point")
    ylabel("Vy")
    ax3.plot(array(range(1, l+1)), yd[:, 1], color = 'r', label = "Vy Data")
    ax3.plot(array(range(1, l+1)), y[:, 1], color = 'b', label = "Vy trained output")
    ax3.legend()
    savefig("q5_3.png")

    #Plot Post processed Weights
    f4 = figure()
    ax4 = f4.add_subplot(1,1,1)
    title("Weights after training (Vx, Vy only; not s_3, s_4 and s_5)")
    ax4.scatter(w[:,3],w[:,4])
    savefig("q5_4.png")

    #Plot error vs epoch
    f7 = figure()
    ax7 = f7.add_subplot(1,1,1)
    title("Question 5: Error Vs. Epoch")
    xlabel("Epoch")
    ylabel("Error")
    ax7.plot(range(1, len(error) + 1), error, color = 'r')
    savefig("q5_7.png")

l, m = shape(testData)
x0 = testData[:,2:m] #clustering data
yd = testData[:,:2] #output data

print len(x0), l, m

y_tmp = np.zeros((l,2))

for j in range(l):
    #loop to compute distance vector for each neuron
    for i in range(sz):
        d[i] = dist(w[i], x0[j])

    #index of winning neuron
    i = argmin(d)

    for k in range(sz):
        #distance from winning neuron
        nd[k] = abs(k - i)

    h = gaussian(n, nd)
    s = sum(h)

    #compute output for each input data
    y_tmp[j] = sum(h*(yLambda + dot((x0[j] - w), aLambda.T)/s), axis=0)

f5 = figure()
ax5 = f5.add_subplot(1,1,1)
title("Question 5: Vx tested")
xlabel("Data Point")
ylabel("Vx")
print len(y_tmp)
print len(yd)
ax5.plot(array(range(l)), yd[:, 0], color = 'r', label = "Vx Data")
ax5.plot(array(range(l)), y_tmp[:, 0], color = 'b', label = "Vx tested output")
ax5.legend()
savefig("q5_5.png")

f6 = figure()
ax6 = f6.add_subplot(1,1,1)
title("Question 5: Vy tested")
xlabel("Data Point")
ylabel("Vy")
ax6.plot(array(range(l)), yd[:, 1], color = 'r', label = "Vy Data")
ax6.plot(array(range(l)), y_tmp[:, 1], color = 'b', label = "Vy tested output")
ax6.legend()
savefig("q5_6.png")
