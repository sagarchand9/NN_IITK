function [weights, neighbour_fun] = update_weights(X, weights, sigma)
    
    epsilon = 0.1;
    [temp1_waste, winner_index, wastes_distances] = distance_cal(X, weights, sigma, epsilon);

    [distances, temp2_waste, neighbour_fun] = distance_cal(weights(winner_index, :), weights, sigma, epsilon);
    
    %neighbour_fun = exp(-(distances.*distances)/(2*sigma*sigma + epsilon));
    
    for j =1:length(weights)
       weights(j, :) = weights(j, :) + 0.01*neighbour_fun(j)*(X - weights(j, :));
    end

end

