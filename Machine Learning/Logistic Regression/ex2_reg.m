%% Regularized Logistic Regression 
% For Project2 using data 2
% Classification with High polynomial Boundary X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..........up to degree 6
% Change lambda to see over-fitting or under-fitting

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column contains the label (y).
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);
hold on;
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')
hold off;

%% =========== Part 1: Regularized Logistic Regression ============

% Add Polynomial Features
% Note that mapFeature also adds a column of ones , so the intercept term is handled

X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Cost Function Test
% Compute and display initial cost and gradient for regularized logistic regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

% Cost Function Test
test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');


%% ============= Part 2: Regularization and Accuracies =============
%  try different values of lambda
%  Try the following values of lambda (0, 1, 10, 100).
%
%  the decision boundary and the training set accuracy change when  vary lambda

% Initialize fitting parameters
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

% Plot Boundary
plotDecisionBoundary(theta, X, y);

hold on;
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');

