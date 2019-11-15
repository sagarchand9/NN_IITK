from __future__ import division
import pandas as pd
import numpy as np
import math
from random import randint
import matplotlib.pyplot as plt


I = 10
J = 10
N = 30000
sigma_0 = 1
L_0 = 0.05
# lamda = float(N)/math.log(sigma_0)
lamda = 10

#y_lambda = np.random.uniform(-1,1,(height, width, output_dim))
#a_lambda = np.random.uniform(-1,1,(height, width, output_dim, input_dim))
#w_lambda = np.random.uniform(-1,1,(height, width, input_dim))

E_x = np.zeros(N)
E_y = np.zeros(N)
E = np.zeros(N)
V_xd = np.zeros(N)
V_yd = np.zeros(N)
V_x = np.zeros(N)
V_y = np.zeros(N)


# reading data
data = pd.read_csv('new_data.txt', sep=" ", header=None)
data.drop(data.index[1], axis=1, inplace=True)
data.drop(data.index[3], axis=1, inplace=True)
data.drop(data.index[5], axis=1, inplace=True)
data.drop(data.index[7], axis=1, inplace=True)
data.drop(data.index[9], axis=1, inplace=True)
data.columns = [0,1,2,3,4]

#output = data.iloc[:,0:2].copy()
#data = data.values

'''def BMU(a1,a2,a3,b1,b2,b3):
	return math.sqrt(pow((a1-b1),2) + pow((a2-b2),2) + pow((a3-b3),2) )'''
	
def BMU(a1,a2,a3,a4,a5,b1,b2,b3,b4,b5):
	return math.sqrt(pow((a1-b1),2) + pow((a2-b2),2) + pow((a3-b3),2) + pow((a4-b4),2) + pow((a5-b5),2))
	
# creating and initialising random weight matrix 
W = np.random.rand(I,J,5) 
W = W*2 - 1
Theta = np.random.rand(I,J,2)
Theta = Theta*2 - 1
A = np.random.rand(I,J,2,5) 
A = A*2 - 1

h = np.zeros((I,J))

# Draw a sample x
for t in range(N):
	n = randint(2, 4997)
	V_xd[t] = float(data.iloc[n][0])
	V_yd[t] = float(data.iloc[n][1])
	y_d = np.array((float(data.iloc[n][0]), float(data.iloc[n][1])))
	u = np.array((float(data.iloc[n][0]), float(data.iloc[n][1]), float(data.iloc[n][2]), float(data.iloc[n][3]), float(data.iloc[n][4])))

	# Similarity matching OR Finding minimum distance
	dist_min = BMU(W[0,0,0], W[0,0,1], W[0,0,2], W[0,0,3], W[0,0,4], float(data.iloc[n][0]), float(data.iloc[n][1]), float(data.iloc[n][2]), float(data.iloc[n][3]), float(data.iloc[n][4]))
	i_min = 0
	j_min = 0

	for i in range(I):
		for j in range(J):
			dist = BMU(W[i,j,0], W[i,j,1], W[i,j,2], W[i,j,3], W[i,j,4], float(data.iloc[n][0]), float(data.iloc[n][1]), float(data.iloc[n][2]), float(data.iloc[n][3]), float(data.iloc[n][4]))
			if(dist < dist_min):
				dist = dist_min
				i_min = i
				j_min = j

	# Computing sigma/radius
	sigma = sigma_0*(math.exp(-(float(t)/float(lamda +  0.000001))))

	# Learning rate
	L = L_0*(math.exp(-(float(t)/float(lamda+ 0.000001))))
	# L=L_0
	# L = L_0*(10*(N-t))/N

	sum_h = 0
	sum_theta = np.zeros(2)
	for i in range(I):
		for j in range(J):
		
			# h
			d2 = pow((i-i_min),2) + pow((j-j_min),2)
			h[i,j] = (math.exp(-(float(d2)/float(2*sigma*sigma +  0.000001))))
			sum_h = sum_h + h[i,j]
			sum_theta = sum_theta + h[i,j]*(Theta[i,j] + A[i,j].dot(W[i,j]))

	y = sum_theta/sum_h
	V_x[t] = y[0]
	V_y[t] = y[1]	
	
	for i in range(I):
		for j in range(J):
			# Update rule
			Theta[i,j] = Theta[i,j] + ((L*h[i,j]*(y_d-y))/sum_h)
			A[i,j,0] = A[i,j,0] + L*h[i,j]*(y_d[0]-y[0])*(u-W[i,j])
			A[i,j,1] = A[i,j,1] + L*h[i,j]*(y_d[1]-y[1])*(u-W[i,j])
			W[i,j] = W[i,j] + L*h[i,j]*(u-W[i,j])

	E_x[t] = (pow((y_d[0]-y[0]),2))/2
	E_y[t] = (pow((y_d[1]-y[1]),2))/2
	E[t] = E_x[t] + E_y[t]
	if(t%10==0):
		print(t)
		print(E_x[t], E_y[t], E[t])
		print(V_xd[t], V_x[t], V_yd[t], V_y[t])
		

x = np.zeros(N)
for i in range(N):
	x[i] = i

# plotting the points 
plt.plot(x, E, label = "E")
plt.plot(x, E_x, label = "E_x")
plt.plot(x, E_y, label = "E_y")
plt.legend()
# function to show the plot


f = plt.figure()
plt.plot(x, V_xd, label = "V_xd")
plt.plot(x, V_x , label = "V_x")
plt.legend()

f1 = plt.figure()
plt.plot(x, V_yd, label = "V_yd")
plt.plot(x, V_y , label = "V_y")

plt.legend()
plt.show()

np.save('Wa', W)
np.save('Thetaa', Theta)
np.save('Aa', A)
'''
1 Each node''s weights are initialized.

2 A vector is chosen at random from the set of training data and presented to the lattice.

3 Every node is examined to calculate which one''s weights are most like the input vector. The winning node is commonly known as the Best Matching Unit (BMU).

4 The radius of the neighbourhood of the BMU is now calculated. This is a value that starts large, typically set to the 'radius' of the lattice,  but diminishes each time-step. Any nodes found within this radius are deemed to be inside the BMU''s neighbourhood.

5 Each neighbouring node''s (the nodes found in step 4) weights are adjusted to make them more like the input vector. The closer a node is to the BMU, the more its weights get altered.

6 Repeat step 2 for N iterations.
'''


 
