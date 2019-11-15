from __future__ import division
import numpy as np
import random
import csv
import ast

def generate(y, u):
	new_y = (y/(1+(y*y))) + (u*u*u)
	return new_y

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def mod_sigmoid(x):
	return (3 / (1 + np.exp(-x))) - 1.5

def findDist(x1, y1, x2, y2, sigmoid=1):
	return np.exp(-((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)) / (2*sigmoid*sigmoid))

def updateParameter(sigma, n, t):
	return (sigma * np.exp(-(n/t)))

data = []
output = []

#----------------------------Reading train data--------------------------------------
count = 0
with open('../data.csv', 'r') as csvfile:
	all_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in all_data:
		count += 1
	print count

with open('../data.csv', 'r') as csvfile:
	tmp = 0
	all_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in all_data:
		# print "yoooo"
		if tmp > 0:
			# print "row: "
			if tmp < count-1:
				data.append([float(row[0]), float(row[2]), float(row[4]), float(row[6]), float(row[8])])
			if tmp > 1:
				output.append([float(row[0]), float(row[2])])
		tmp += 1

#-----------------------------------------------------------------------------------

data = np.asarray(data)
output = np.asarray(output)
print data.shape, output.shape

height = 10
width = 10
output_dim = 2
input_dim = 5
lr_w = 0.05
lr_y = 0.05
lr_a = 0.05
sigma = 1.0
time_constant_width = 10
time_constant_learning = 10

y_lambda = np.random.uniform(-1,1,(height, width, output_dim))
a_lambda = np.random.uniform(-1,1,(height, width, output_dim, input_dim))
w_lambda = np.random.uniform(-1,1,(height, width, input_dim))

#--------------------train-----------------------------------------------------------
for epoch in range(1000000):
	# i = random.randint(1,data.shape[0]-1)
	error = 0.0
	for i in range(data.shape[0]):
		inp = data[i]
		out = output[i]
		inp = inp.reshape(input_dim,1)
		out = out.reshape(output_dim,1)
		winning_neuron = (0, 0)
		cur_w = w_lambda[0][0]
		cur_w = cur_w.reshape(input_dim, 1)
		min_dist = np.linalg.norm(inp - cur_w)
		for j in range(height):
			for k in range(width):
				cur_w = w_lambda[j][k]
				cur_w = cur_w.reshape(input_dim, 1)
				cur_dist = np.linalg.norm(inp - cur_w)
				# print j, k, cur_dist, min_dist
				if cur_dist < min_dist:
					winning_neuron = (j, k)
					min_dist = cur_dist
		# print i, winning_neuron
		h_lambda = np.zeros((height, width))
		s = 0.0
		for j in range(height):
			for k in range(width):
				h_lambda[j][k] = findDist(j, k, winning_neuron[0], winning_neuron[1], sigma)
				s += h_lambda[j][k]
		# print h_lambda[0][0], winning_neuron
		# print s
		y_pred = np.zeros((output_dim, 1))
		for j in range(height):
			for k in range(width):
				cur_y = y_lambda[j][k]
				cur_a = a_lambda[j][k]
				cur_w = w_lambda[j][k]
				cur_y = cur_y.reshape(output_dim, 1)
				cur_a = cur_a.reshape(output_dim, input_dim)
				cur_w = cur_w.reshape(input_dim, 1)
				y_pred += h_lambda[j][k] * (cur_y + np.matmul(cur_a, inp-cur_w))
				# tmp = cur_a * (inp-cur_w)
		y_pred = y_pred / s
		# print y_pred #2x1
		# print "initial: ", w_lambda[0][0]
		for j in range(height):
			for k in range(width):
				#-------------------update-w-------------------------------
				cur_w = w_lambda[j][k]
				cur_w = cur_w.reshape(input_dim, 1)
				tmp = lr_w * h_lambda[j][k] * (inp - cur_w)
				tmp = tmp.reshape(input_dim)
				w_lambda[j][k] += tmp
				#----------------------------------------------------------
				# print "middle: ", w_lambda[0][0], tmp

				#-------------------update-y-------------------------------
				tmp = (lr_y/s) * h_lambda[j][k] * (out - y_pred)
				tmp = tmp.reshape(output_dim)
				y_lambda[j][k] += tmp
				#----------------------------------------------------------

				#-------------------update-a-------------------------------
				cur_w = w_lambda[j][k]
				cur_w = cur_w.reshape(input_dim, 1)
				tmp = (lr_a/s) * h_lambda[j][k] * np.matmul(out - y_pred, np.transpose(inp - cur_w))
				tmp = tmp.reshape(output_dim, input_dim)
				a_lambda[j][k] += tmp
				#----------------------------------------------------------
		error += 0.5 * np.linalg.norm(out - y_pred) * np.linalg.norm(out - y_pred)
	# sigma = updateParameter(sigma, epoch+1, time_constant_width)
	# lr_w = updateParameter(lr_w, epoch+1, time_constant_learning)
	# lr_y = updateParameter(lr_y, epoch+1, time_constant_learning)
	# lr_a = updateParameter(lr_a, epoch+1, time_constant_learning)
	print "Epoch: " + str(epoch) + " | Error: " + str(error/data.shape[0])
	np.save('w_lambda', w_lambda)
	np.save('y_lambda', y_lambda)
	np.save('a_lambda', a_lambda)

# np.save('w_lambda', w_lambda)
# np.save('y_lambda', y_lambda)
# np.save('a_lambda', a_lambda)
#--------------------train-----------------------------------------------------------

#--------------------test-----------------------------------------------------------

