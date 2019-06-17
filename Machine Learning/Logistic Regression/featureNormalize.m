function [X_norm, mu, sigma] = featureNormalize(X)
% X_norm = X afer normalize
% mu = average of X for each column
% sigma = standard deviation of X for each column 
mu = mean(X);
X_norm = X - mu;
sigma = std(X_norm);
X_norm = X_norm./sigma;


end