function [w_infer,y] = rvm_train(ds,is,width)
% <============ HEADER =============>
% @brief    : train a RVM model 
% @params   : ds <- dataset
%             is <- input vectors dimensionality
%             width <- RBF kernel width
% @returns  : 
% <============ HEADER =============>

n = size(ds,1);
inputs = ds(:,1:is);
targets = ds(:,is+1);
BASIS	= exp(-distSquared(inputs,inputs)/(width^2));
BASIS = [BASIS,ones(n,1)];
sig = @(x) 1/(1+exp(-x));


% Training
[Parameter, Hyperparameter, Diagnostic] = ...
    SparseBayes('Bernoulli', BASIS, targets);
w_infer						= zeros(n+1,1);
w_infer(Parameter.Relevant)	= Parameter.Value;

% Predictions 
y = arrayfun(sig,BASIS*w_infer);
%y = double(y>=0.5);
end


function D2 = distSquared(X,Y)
%
nx	= size(X,1);
ny	= size(Y,1);
%
D2 = (sum((X.^2), 2) * ones(1,ny)) + (ones(nx, 1) * sum((Y.^2),2)') - ...
     2*X*Y';
end
