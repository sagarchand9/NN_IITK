function [distance, index, neighbour_fun] = distance_cal(X_temp, weights, sigma, epsilon)

    for j=1:100
        distance(j) = norm(X_temp - weights(j, :));
    end
    
    [value, index] = min(distance);
    neighbour_fun = exp(-(distance.*distance)/(2*sigma*sigma + epsilon));
end