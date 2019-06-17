function plotDecisionBoundary(theta, X, y)
%   Plots the data points X and y into a new figure with the decision boundary defined by theta
%   + for the  positive examples and o for the negative examples.  
%   X is assumed to be a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3  
    % if less than three features (include x0)
    % Decision boundary  is theta0 + theta1*x1+theta2*x2 = 0
  
    % Only need 2 points to define a line, so choose two endpoints
    % two points' x value in axis x1
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate two points' y value in axis x2 for the decision boundary line
    % x2 = -(theta1*x1+theta0)/theta2
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
    %axis([-1, 1.5, -1, 1.5])
else
    % if greater than three features (means high polynomial)
    % Here is the axis range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end
