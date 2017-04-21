function [wMap,Sn] = laplax(ds, is)
% <============ HEADER =============>
% @brief    : computes the posterior approximation with Laplace
%             approximation
%             feature space is input space 
% @params   : ds <- dataset
%             is <- input vectors dimensionality
% @returns  : posterior distribution q
% <============ HEADER =============>

mo = zeros(is+1,1);
So = 10*eye(is+1);
prior = gaussianDb(is,mo,So); % start with centered Gaussian prior 
outp  = @(w) compute_output('logistic_sigmoid',w(1:is),w(is+1),ds(:,1:is),'linear');

% Finding the posterior MAP value
max_iter = 10;
lr = 1;
log_posterior = @(x) log(prior(x)) + cross_entropy_loss_function(outp(x),ds(:,is+1));
w = zeros(is+1,1);
eps = 0.001;
for i=1:max_iter
   X = [ds(:,1:is),ones(size(ds,1),1)]; y = outp(w); t = ds(:,is+1);    % design - pred - labels 
   H = X'*diag(y.*(1-y))*X;                                             % full hessian
   g = So\(w-mo) -X'*(t-y);                                             % gradient
   d = (inv(So)+H)\g;
   w = w - lr*d;
   if (abs(lr*d)<eps)
       break;                                   % stopping criterion 
   end
end

% Estimating the Hessian of log(posterior) at MAP value 
X = [ds(:,1:is),ones(size(ds,1),1)]; y = outp(w);
Sn = inv(inv(So) + X'*diag(y.*(1-y))*X);
wMap = w;


end