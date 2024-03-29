function p = predict(theta, X)
% Predict whether the label is 0 or 1 using learned theta
%   Using a threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

p = zeros(m, 1);

p = sigmoid(X*theta);

for i = 1:m
  if p(i)>=0.5
    p(i) =  1;
  else
    p(i) = 0;    
  end
end

%p(find(p>=0.5)) = 1;
%p(find(p<0.5)) = 0;

end
