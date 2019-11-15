Author: Pawan Harendra Mishra
EE671A: Neural Networks

***********************
data.csv
***********************
data1.csv and data2.csv are essentially same
data1.csv can be read by numpy whereas data2.csv is raw data csv file
Contains comma seperated training data.
The fields are:
    1. Vx
    2. Vy
    3. s_3
    4. s_4
    5. s_5
    6. previous Vx value
    7. previous Vy value
Note: initially, previous Vx and previous Vy are zero

***********************
Question 1:
***********************
Backpropagation with Momentum
Five input nodes:
    1. s_3
    2. s_4
    3. s_5
    4. previous Vx
    5. previous Vy
Two output nodes:
    1. Vx
    2. Vy
This is a generalized code.
It can be easily modified to work with more number of hidden layers.
Number of nodes in each hidden layer can be choosen by changing global parameters.
Only partial testing, optimization and tuning of parameters is complete.

***********************
Question 2:
***********************
Backpropagation with Adaptive Learning
Five input nodes:
    1. s_3
    2. s_4
    3. s_5
    4. previous Vx
    5. previous Vy
Two output nodes:
    1. Vx
    2. Vy
This is a generalized code.
It can be easily modified to work with more number of hidden layers.
Number of nodes in each hidden layer can be choosen by changing global parameters.
Only partial testing, optimization and tuning of parameters is complete.

***********************
Question 5:
***********************
Clustering by 1-dimensional self organizing map
Output by collective response model
Input vector:
    1. s_3
    2. s_4
    3. s_5
    4. previous Vx
    5. previous Vy
Output vector:
    1. Vx
    2. Vy
This is a generalized code.
The number of neurons and other parameters can be chosen by changing global parameters.
Testing, optimization and tuning of parameters is incomplete.
