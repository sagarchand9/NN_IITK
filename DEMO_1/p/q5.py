from numpy import *
from matplotlib.pyplot import *

#training data from csv file
data = genfromtxt("data1.csv", delimiter = ',')

#global parameters
l, m = shape(data)
lr0 = 0.01 #learning rate parameter for weight
#yLambda and aLambda are initialized below
lr1 = 0.01 #learning rate parameter for yLambda
lr2 = 0.01 #learning rate parameter for aLambda
t1 = 10 #time constant of learning rate
t2 = 10 #time constant of width function
sig0 = 1 #initial value of width
sz = 25 #size of 1-D SOM
maxIt = 1000 #maximum training steps

x0 = data[:,2:m] #clustering data
yd = data[:,:2] #output data

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
    aLambda = random.random((sz,1))

    #plot initialized weights
    f1 = figure()
    ax1 = f1.add_subplot(1,1,1)
    title("Weights before training (Vx, Vy only; not s_3, s_4 and s_5)")
    ax1.scatter(w[:,3], w[:,4])
    savefig("q5_1.png")

    d = empty(sz) #weight-input distance array
    nd = empty((sz,1)) #neuron distance array
    n = 0 #training steps

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
            y = sum(h*(yLambda + aLambda*(x0[j, 3:] - w[:, 3:])/s), axis=0)

            e = yd[j] - y

            for k in range(sz):
                #weight update
                w[k] = w[k] + neta*h[k]*(x0[j] - w[k])

            #yLambda and aLambda update
            yLambda = yLambda + lr1*e*h/s
            aLambda = aLambda + lr2*e*(x0[j, 3:] - w[:, 3:])/s

        print n, sqrt(sum(e**2))
        if n >= maxIt:
            break

    return w, yLambda, aLambda


w, yLambda, aLambda = clustering()
savetxt("q5_w.csv", w, delimiter = ',')
savetxt("q5_yLambda.csv", yLambda, delimiter = ',')
savetxt("q5_aLambda.csv", aLambda, delimiter = ',')

y = empty((l,2)) #output arrays
d = empty(sz) #weight-input distance array
nd = empty((sz,1)) #neuron distance array
n = 1

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

for i in range(l):
    #compute output for each input data
    y[i] = sum(h*(yLambda + aLambda*(x0[i, 3:] - w[:, 3:])/s), axis=0)

#plot output and actual data
f2 = figure()
ax2 = f2.add_subplot(1,1,1)
title("Question 5: Vx")
xlabel("Data Point")
ylabel("Vx")
ax2.plot(array(range(1, l+1)), yd[:, 0], color = 'r', label = "Vx Data")
ax2.plot(array(range(1, l+1)), y[:, 0], color = 'b', label = "Vx Output")
ax2.legend()
savefig("q5_2.png")

f3 = figure()
ax3 = f3.add_subplot(1,1,1)
title("Question 5: Vy")
xlabel("Data Point")
ylabel("Vy")
ax3.plot(array(range(1, l+1)), yd[:, 1], color = 'r', label = "Vy Data")
ax3.plot(array(range(1, l+1)), y[:, 1], color = 'b', label = "Vy Output")
ax3.legend()
savefig("q5_3.png")

#analyse clusters
neuronClustering = zeros((sz,1), dtype=int)
clusterSum = zeros((sz, 2))
d = empty(sz)

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


#Plot Post processed Weights
f4 = figure()
ax4 = f4.add_subplot(1,1,1)
title("Weights after training (Vx, Vy only; not s_3, s_4 and s_5)")
ax4.scatter(w[:,3],w[:,4])
savefig("q5_4.png")
