from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import math

l1 = 0.254
l2 = 0.254
l3 = 0.254
t = 0.050

def gauss(inp, c, sig):
	z = np.linalg.norm(inp-c)
	# print "z: ", z
	phi = math.exp(-(z*z)/(2*sig*sig))
	# print phi
	return phi

def theta_values(x):
	# print "x: ", x
	a = math.sqrt(x[0]*x[0] + x[1]*x[1]) - t
	b = x[2] - l1
	K = b*b + l2*l2 - l3*l3 + a*a
	A = 4*a*a*l2*l2 + 4*b*b*l2*l2
	B = -4*a*l2*K
	C = K*K - 4*b*b*l2*l2
	x_new = (-B + math.sqrt(B*B - 4*A*C)) / (2*A)
	# print "x_new: ", x_new
	theta1 = math.atan(x[1] / x[0])
	theta2 = math.acos(x_new)
	theta3 = math.asin((b - l2*math.sin(theta2)) / l3)
	return [theta1, theta2, theta3]

def coordinate_from_angle(theta):
	r = l2*math.cos(theta[1]) + l3*math.cos(theta[2]) + t
	x1 = r*math.cos(theta[0])
	x2 = r*math.sin(theta[0])
	x3 = l2*math.sin(theta[1]) + l3*math.sin(theta[2]) + l1
	return [x1, x2, x3]

theta1_min = -160
theta1_max = 160
theta2_min = -120
theta2_max = 120
theta3_min = -720
theta3_max = 720

train_size = 100
input_dim = 3
output_dim = 3
k_new = 1


#--------------------------train data generation----------------------------------------
train_data_x_start = []
train_data_theta_start = []
train_data_x_desired = []
for i in range(train_size):
	#------------------initial x-------------------------------------
	theta1 = math.radians(random.randint(theta1_min, theta1_max))
	theta2 = math.radians(random.randint(theta2_min, theta2_max))
	theta3 = math.radians(random.randint(theta3_min, theta3_max))
	# theta1 = 0.0
	# theta2 = 0.0
	# theta3 = 0.0
	r = l2*math.cos(theta2) + l3*math.cos(theta3) + t
	x = r*math.cos(theta1)
	y = r*math.sin(theta1)
	z = l2*math.sin(theta2) + l3*math.sin(theta3) + l1	
	train_data_x_start.append([x, y, z])
	train_data_theta_start.append([theta1, theta2, theta3])
	#----------------------------------------------------------------

	#----------------desired x---------------------------------------
	theta1 = math.radians(random.randint(theta1_min, theta1_max))
	theta2 = math.radians(random.randint(theta2_min, theta2_max))
	theta3 = math.radians(random.randint(theta3_min, theta3_max))
	r = l2*math.cos(theta2) + l3*math.cos(theta3) + t
	x = r*math.cos(theta1)
	y = r*math.sin(theta1)
	z = l2*math.sin(theta2) + l3*math.sin(theta3) + l1	
	train_data_x_desired.append([x, y, z])
	#----------------------------------------------------------------
#---------------------------------------------------------------------------------------

rbfn_centres = 10
rbfn_lr = 0.05
rbfn_sigma = 1
rbfn_c = random.sample(train_data_x_desired, rbfn_centres)
rbfn_c = np.array(rbfn_c) #10x3
rbfn_w = np.random.uniform(-1,1,(rbfn_centres, output_dim)) #10x3

R = np.identity(input_dim, dtype="int")
Q = np.identity(input_dim, dtype="int")

for n in range(train_size):
	k = 0
	x = train_data_x_start[n] #x_start
	x = np.array(x)
	x = x.reshape(input_dim) #robot is initially at this position
	theta = train_data_theta_start[n] #theta_start
	theta = np.array(theta)
	x_d = train_data_x_desired[n]
	x_d = np.array(x_d)
	x_d = x_d.reshape(input_dim) #desired position
	while True:
		e = x_d - x #e(k)

		#--------------------------------critic network(RBFN)-------------------------------------------------
		phi = []
		for j in range(rbfn_centres):
			phi.append(gauss(k_new*e, rbfn_c[j], rbfn_sigma))
		phi = np.array(phi)
		phi = phi.reshape(1, rbfn_centres)
		lambda_0 = np.matmul(phi, rbfn_w) #1x3 lambda(k+1)
		#-----------------------------------------------------------------------------------------------------
		r = l2*math.cos(theta[1]) + l3*math.cos(theta[2]) + t
		jacobian = np.zeros((input_dim, output_dim))
		jacobian[0][0] = -r*math.sin(theta[0])
		jacobian[1][0] = r*math.cos(theta[0])
		jacobian[2][0] = 0
		jacobian[0][1] = -l2*math.sin(theta[1])*math.cos(theta[0])
		jacobian[1][1] = -l2*math.sin(theta[1])*math.sin(theta[0])
		jacobian[2][1] = l2*math.cos(theta[1])
		jacobian[0][2] = -l3*math.sin(theta[2])*math.cos(theta[0])
		jacobian[1][2] = -l3*math.sin(theta[2])*math.sin(theta[0])
		jacobian[2][2] = l3*math.cos(theta[2])

		#------------------find delta_theta -> find theta --> find x(k+1)----------
		# delta_theta = np.matmul(np.matmul(np.linalg.inv(R), np.transpose(jacobian)), np.transpose(lambda_0)) #3x1
		# theta[0] += delta_theta[0][0]
		# theta[1] += delta_theta[1][0]
		# theta[2] += delta_theta[2][0]

		# r = l2*math.cos(theta[1]) + l3*math.cos(theta[2]) + t
		# x[0] = r*math.cos(theta[0])
		# x[1] = r*math.sin(theta[0])
		# x[2] = l2*math.sin(theta[1]) + l3*math.sin(theta[2]) + l1
		#---------------------------------------------------------------------------

		#----------------find delta_theta -> find x(k+1) -> find theta-------------- 
		# delta_theta = np.matmul(np.matmul(np.linalg.inv(R), np.transpose(jacobian)), np.transpose(lambda_0)) #3x1
		# tmp = np.matmul(jacobian, delta_theta) #3x1
		# x[0] += tmp[0][0]
		# x[1] += tmp[1][0]
		# x[2] += tmp[2][0]

		# tmp = theta_values(x)
		# theta[0] = tmp[0]
		# theta[1] = tmp[1]
		# theta[2] = tmp[2]
		#----------------------------------------------------------------------------

		#-------------find delt_theta -> find x(k+1) and theta-----------------------
		delta_theta = np.matmul(np.matmul(np.linalg.inv(R), np.transpose(jacobian)), np.transpose(lambda_0)) #3x1
		tmp = np.matmul(jacobian, delta_theta) #3x1
		x[0] += tmp[0][0]
		x[1] += tmp[1][0]
		x[2] += tmp[2][0]

		theta[0] += delta_theta[0][0]
		theta[1] += delta_theta[1][0]
		theta[2] += delta_theta[2][0]
		#-----------------------------------------------------------------------------


		e_new = x_d - x #e(k+1)
		#--------------------------------critic network(RBFN)-------------------------------------------------
		phi_new = []
		for j in range(rbfn_centres):
			phi_new.append(gauss(k_new*e_new, rbfn_c[j], rbfn_sigma))
		phi_new = np.array(phi)
		phi_new = phi_new.reshape(1, rbfn_centres)
		lambda_1 = np.matmul(phi_new, rbfn_w) #1x3 lambda(k+2)
		#-----------------------------------------------------------------------------------------------------
		e_new = e_new.reshape(input_dim, 1) #3x1
		lambda_desired_0 = np.matmul(np.transpose(e_new), Q) + lambda_1 #1x3 lambda_d(k+1)
		#------------------------------backprop in RBFN-------------------------------------------------------
		#----------------weight update------------------------------------
		for j in range(rbfn_centres):
			tmp = rbfn_lr*phi[0][j]*(lambda_desired_0 - lambda_0) #1x3
			rbfn_w[j][0] += tmp[0][0]
			rbfn_w[j][1] += tmp[0][1]
			rbfn_w[j][2] += tmp[0][2]
		#-----------------------------------------------------------------
		#----------------centre update------------------------------------
		for j in range(rbfn_centres):
			tmp = rbfn_w[j] #3D
			tmp = tmp.reshape(input_dim, 1) #3x1
			for l in range(input_dim):
				rbfn_c[j][l] += rbfn_lr*phi[0][j]*(e[l]-rbfn_c[j][l])*np.matmul(lambda_desired_0 - lambda_0, tmp)
		#-----------------------------------------------------------------
		# print k, " | ", np.linalg.norm(e)
		# print rbfn_w
		k += 1
		if np.linalg.norm(e) < 0.0001 or k > 1000:
			print n, k
			break



#--------------------------test data generation----------------------------------------
plt_x_predicted = []
plt_x_desired = []
plt_y_predicted = []
plt_y_desired = []
plt_z_predicted = []
plt_z_desired = []
test_size = 100
test_data_x_start = []
test_data_theta_start = []
test_data_x_desired = []
min_x = -0.3
max_x = 0.3

theta1 = math.radians(random.randint(theta1_min, theta1_max))
theta2 = math.radians(random.randint(theta2_min, theta2_max))
theta3 = math.radians(random.randint(theta3_min, theta3_max))
r = l2*math.cos(theta2) + l3*math.cos(theta3) + t
x = r*math.cos(theta1)
y = r*math.sin(theta1)
z = l2*math.sin(theta2) + l3*math.sin(theta3) + l1	
test_data_x_start.append([x, y, z])
test_data_theta_start.append([theta1, theta2, theta3])

toadd = (max_x - min_x) / test_size

#------------------------------line------------------------------------------
x = min_x
y = (5/6) * x + (11/20)
z = (5/6) * (x + 0.3)
test_data_x_desired.append([x, y, z])
#------------------------------line------------------------------------------

for n in range(test_size):
	k = 0
	x = test_data_x_start[n] #x_start
	x = np.array(x)
	x = x.reshape(input_dim) #robot is initially at this position
	theta = test_data_theta_start[n] #theta_start
	theta = np.array(theta)
	x_d = test_data_x_desired[n]
	x_d = np.array(x_d)
	x_d = x_d.reshape(input_dim) #desired position
	while True:
		e = x_d - x #e(k)

		#--------------------------------critic network(RBFN)-------------------------------------------------
		phi = []
		for j in range(rbfn_centres):
			phi.append(gauss(k_new*e, rbfn_c[j], rbfn_sigma))
		phi = np.array(phi)
		phi = phi.reshape(1, rbfn_centres)
		lambda_0 = np.matmul(phi, rbfn_w) #1x3 lambda(k+1)
		#-----------------------------------------------------------------------------------------------------
		r = l2*math.cos(theta[1]) + l3*math.cos(theta[2]) + t
		jacobian = np.zeros((input_dim, output_dim))
		jacobian[0][0] = -r*math.sin(theta[0])
		jacobian[1][0] = r*math.cos(theta[0])
		jacobian[2][0] = 0
		jacobian[0][1] = -l2*math.sin(theta[1])*math.cos(theta[0])
		jacobian[1][1] = -l2*math.sin(theta[1])*math.sin(theta[0])
		jacobian[2][1] = l2*math.cos(theta[1])
		jacobian[0][2] = -l3*math.sin(theta[2])*math.cos(theta[0])
		jacobian[1][2] = -l3*math.sin(theta[2])*math.sin(theta[0])
		jacobian[2][2] = l3*math.cos(theta[2])

		#------------------find delta_theta -> find theta --> find x(k+1)----------
		# delta_theta = np.matmul(np.matmul(np.linalg.inv(R), np.transpose(jacobian)), np.transpose(lambda_0)) #3x1
		# theta[0] += delta_theta[0][0]
		# theta[1] += delta_theta[1][0]
		# theta[2] += delta_theta[2][0]

		# r = l2*math.cos(theta[1]) + l3*math.cos(theta[2]) + t
		# x[0] = r*math.cos(theta[0])
		# x[1] = r*math.sin(theta[0])
		# x[2] = l2*math.sin(theta[1]) + l3*math.sin(theta[2]) + l1
		#---------------------------------------------------------------------------

		#----------------find delta_theta -> find x(k+1) -> find theta-------------- 
		# delta_theta = np.matmul(np.matmul(np.linalg.inv(R), np.transpose(jacobian)), np.transpose(lambda_0)) #3x1
		# tmp = np.matmul(jacobian, delta_theta) #3x1
		# x[0] += tmp[0][0]
		# x[1] += tmp[1][0]
		# x[2] += tmp[2][0]

		# tmp = theta_values(x)
		# theta[0] = tmp[0]
		# theta[1] = tmp[1]
		# theta[2] = tmp[2]
		#----------------------------------------------------------------------------

		#-------------find delt_theta -> find x(k+1) and theta-----------------------
		delta_theta = np.matmul(np.matmul(np.linalg.inv(R), np.transpose(jacobian)), np.transpose(lambda_0)) #3x1
		tmp = np.matmul(jacobian, delta_theta) #3x1
		x[0] += tmp[0][0]
		x[1] += tmp[1][0]
		x[2] += tmp[2][0]

		theta[0] += delta_theta[0][0]
		theta[1] += delta_theta[1][0]
		theta[2] += delta_theta[2][0]
		#-----------------------------------------------------------------------------


		e_new = x_d - x #e(k+1)
		#--------------------------------critic network(RBFN)-------------------------------------------------
		phi_new = []
		for j in range(rbfn_centres):
			phi_new.append(gauss(k_new*e_new, rbfn_c[j], rbfn_sigma))
		phi_new = np.array(phi)
		phi_new = phi_new.reshape(1, rbfn_centres)
		lambda_1 = np.matmul(phi_new, rbfn_w) #1x3 lambda(k+2)
		#-----------------------------------------------------------------------------------------------------
		e_new = e_new.reshape(input_dim, 1) #3x1
		lambda_desired_0 = np.matmul(np.transpose(e_new), Q) + lambda_1 #1x3 lambda_d(k+1)
		#------------------------------backprop in RBFN-------------------------------------------------------
		#----------------weight update------------------------------------
		for j in range(rbfn_centres):
			tmp = rbfn_lr*phi[0][j]*(lambda_desired_0 - lambda_0) #1x3
			rbfn_w[j][0] += tmp[0][0]
			rbfn_w[j][1] += tmp[0][1]
			rbfn_w[j][2] += tmp[0][2]
		#-----------------------------------------------------------------
		#----------------centre update------------------------------------
		for j in range(rbfn_centres):
			tmp = rbfn_w[j] #3D
			tmp = tmp.reshape(input_dim, 1) #3x1
			for l in range(input_dim):
				rbfn_c[j][l] += rbfn_lr*phi[0][j]*(e[l]-rbfn_c[j][l])*np.matmul(lambda_desired_0 - lambda_0, tmp)
		#-----------------------------------------------------------------
		k += 1
		if np.linalg.norm(e) < 0.0001 or k > 50:
			print n, k, np.linalg.norm(e)
			break

	x_predicted = [x[0], x[1], x[2]]
	plt_x_predicted.append(x_predicted[0])
	plt_y_predicted.append(x_predicted[1])
	plt_z_predicted.append(x_predicted[2])
	plt_x_desired.append(x_d[0])
	plt_y_desired.append(x_d[1])
	plt_z_desired.append(x_d[2])


	test_data_x_start.append([x[0], x[1], x[2]])
	test_data_theta_start.append([theta[0], theta[1], theta[2]])
	#------------------------------line------------------------------------------
	x = x_d[0] + toadd
	y = (5/6) * x + (11/20)
	z = (5/6) * (x + 0.3)
	test_data_x_desired.append([x, y, z])
	#------------------------------line------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(plt_x_desired, plt_y_desired, plt_z_desired, c='r', marker='o')
ax.scatter(plt_x_predicted, plt_y_predicted, plt_z_predicted, c='g', marker='^')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Train v/s Test[Desired in red and Predicted in green]')
# plt.show()
plt.savefig('line.png')