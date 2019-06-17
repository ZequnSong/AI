function [J, grad] = costFunction(theta, X, y)

%   J = COSTFUNCTION(theta, X, y) c
%   Compute cost and set it to J
%   Compute the partial derivatives and set grad to the partial derivatives of the cost 

% Initialize some useful values
m = length(y); % number of training examples

h = sigmoid(X*theta);

J = 1/m*(-y'*log(h)-(1-y)'*log(1-h));

grad = 1/m*X'*(h-y);

end
