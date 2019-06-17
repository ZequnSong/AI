%% Logistic Regression 
% For Project1
% Classification with High polynomial Boundary X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..........up to degree 6
% Change lambda to see over-fitting or under-fitting

%% Initialization
clear ; close all; clc

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

%% ==================== Part 1: Plotting ====================
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

%% ============ Part 2:  Regularized Logistic Regression ============

% if original data varys a lot, it's important to normalize features befroe using high polynomial
[X mu sigma] = featureNormalize(X);

X = mapFeature(X(:,1), X(:,2));

initial_theta = zeros(size(X, 2), 1);

%	controls a trade off between fitting the data well and avoid overfitting
% if lambda is too large, will cause under fitting
% if lambda is too small, will cause over fitting
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

 %% ============ Part 3: Plot Boundary ============
plotData(X(:,2:3), y);
hold on

u = linspace(-2, 2.5, 50);
v = linspace(-2, 2.5, 50);
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

hold on;
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

%% ============ Part 4: Train Accuracy =============
% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');
