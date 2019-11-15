clear all;
clear vars;
clc;

%% data generation

rng(10); 
R = 1;
l = 1;
theta = abs(randn(100,3));
X1 = @(R, var1) (R*cos(var1));
X2 = @(R, var2) (R * sin(var2));
X3 = @(R, var2, var3) (l * (1 + cos(var2 + cos(var3))));

%% 

x = X1(R, theta(:, 1));
y = X2(R, theta(:, 2));
z = X3(R, theta(:, 2), theta(: ,3));
X = [x y z];

%% weights intialization

%rng(10);
weights_target_space = abs(randn(100, 3));
weights_theta_space = abs(randn(100,3));
A = abs(randn(100, 3, 3));

%% train target space

eta_target = 1;
for i=1:100
    
    tau = 100/2;
    sigma = 0.1 + exp(-i/tau);
    temp_index = randi(100);
    X_temp = X(temp_index, :);
    
    weights_target_space = update_weights(X_temp, weights_target_space, sigma);
    
end


%% plotting 3d

%scatter3(weights_target_space(:,1),weights_target_space(:,2),weights_target_space(:,3));

%% start training

for k=1:400
    %% train output space

    temp_index = randi(100);
    target = X(temp_index, :);

    temp_sum = 0;
    sigma_problem = exp(-0.1);
    epsilon = 0.1;
    [distances_wastes, wastes, neighbour_fun] = distance_cal(target, weights_target_space, sigma_problem, epsilon);

    for i=1:length(weights_target_space)
        temp = (target - weights_target_space(i, :));
        temp_theta = weights_theta_space(i, :);
        temp_sum = temp_sum + neighbour_fun(i)*(temp_theta + temp*permute(A(i,:,:),[2 3 1]));
    end
    s = sum(neighbour_fun);
    theta_out_0 = inv(s)* temp_sum;

    %% course and fine action

    v_0 = [X1(R, theta_out_0(1)) X2(R, theta_out_0(2)) X3(R, theta_out_0(2), theta_out_0(3))];

    temp_sum = 0;
    for i=1:length(weights_target_space)
       temp = (target - v_0);
       temp_sum = temp_sum + neighbour_fun(i)*(temp*permute(A(i,:,:),[2,3,1]));
    end
    theta_out_1 = theta_out_0 + inv(s)*temp_sum;
    v_1 = [X1(R, theta_out_1(1)) X2(R, theta_out_1(2)) X3(R, theta_out_0(2), theta_out_1(3))];


    %% predicted theta_out_0 and theta_out_1


    [distances_wastes, wastes, neighbour_fun_pred1] = distance_cal(v_0, weights_target_space, sigma_problem, epsilon);
    temp_sum = 0;
    for i=1:length(weights_target_space)
        temp = (target - weights_target_space(i, :));
        temp_theta = weights_theta_space(i, :);
        temp_sum = temp_sum + neighbour_fun_pred1(i)*(temp_theta + temp*permute(A(i,:,:),[2 3 1]));
    end
    s_pred1 = sum(neighbour_fun_pred1);
    theta_out_0_pred = inv(s_pred1)* temp_sum;

    [distances_wastes, wastes, neighbour_fun_pred2] = distance_cal(v_1, weights_target_space, sigma_problem, epsilon);
    temp_sum = 0;
    for i=1:length(weights_target_space)
       temp = (target - v_0);
       temp_sum = temp_sum + neighbour_fun_pred2(i)*(temp*permute(A(i,:,:),[2,3,1]));
    end
    s_pred2 = sum(neighbour_fun_pred2);
    theta_out_1_pred = inv(s_pred2)*temp_sum;

    %% update theta

    eta = 1;
    for i = 1:length(weights_target_space)
        weights_theta_space(i, :) = weights_theta_space(i, :) + eta*s*neighbour_fun(i)*(theta_out_0 - theta_out_0_pred);
    end

    %% update A matrix
    delta_v = v_1 - v_0;
    delta_theta_out = theta_out_1 - theta_out_0;
    delta_theta_out_pred = theta_out_1_pred - theta_out_0_pred;
    for i = 1:length(weights_target_space)
        temp_A = permute(A(i,:,:),[2 3 1]) + eta*s*inv(sum(norm(delta_v)+1))*neighbour_fun(i)*(delta_theta_out - delta_theta_out_pred)'*(delta_v); 
        A(i, :, :) = temp_A
    end


 
end


%% testing

temp_index = randi(100);
target_theta_test = theta(temp_index, :);

x = X1(R, target_theta_test(:, 1));
y = X2(R, target_theta_test(:, 2));
z = X3(R, target_theta_test(:, 2), target_theta_test(: ,3));
target_test = [x y z];

temp_sum = 0;
sigma_problem = exp(-0.1);
epsilon = 0.1;
[distances_wastes, winner, neighbour_fun_test] = distance_cal(target_test, weights_target_space, sigma_problem, epsilon);

for i=1:1%length(weights_target_space)
    temp = (target_test - weights_target_space(winner, :));
    temp_theta = weights_theta_space(i, :);
    temp_sum = temp_sum + neighbour_fun_test(winner)*(temp_theta + temp*permute(A(winner,:,:),[2 3 1]));
end
s = sum(neighbour_fun_test);
theta_out_test = inv(s)* temp_sum;



