%  Linear regression with multiple variables

%% Clear and Close Figures
clear ; close all; clc

%% Load Data
fprintf('Loading data ...\n');
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

%% ================ Part 1: Feature Normalization ================

% Scale features and set them to zero mean
fprintf('\n Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);
fprintf('First 10 examples from the dataset after scaling: \n');
fprintf(' x = [%.2f %.2f] \n', X(1:10,:)');

%% ================ Part 2: Gradient Descent ================

fprintf('\n Running gradient descent ...\n');

% Add intercept term to X
X = [ones(m, 1) X];
% Choose some alpha value
alpha = 0.1;
num_iters = 50;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph    Cost J - Iteration
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Predict the price of a 1650 sq-ft, 3 br house
% At prediction, make sure you do the same feature normalization.
data_test = [1 ([1650 3]-mu)./sigma];
price = data_test*theta; 

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

%% ================ Part 3: Normal Equations ================

fprintf('\n Solving with normal equations...\n');

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Predict the price of a 1650 sq-ft, 3 br house

price = [1 1650 3]*theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

