function [wMap,Sn] = laplax_normal(ds, is, prior)
% <============ HEADER =============>
% @brief    : computes the posterior approximation with Laplace
%             approximation, assuming Gaussian prior 
%             feature space is input space 
% @params   : ds <- dataset
%             is <- input vectors dimensionality
%             prior <- {mean,covariance matrix} structure -- assuming
%                       Gaussian prior for now 
% @returns  : posterior distribution q
% <============ HEADER =============>

mo = prior.mean;
So = prior.covmat;
prior = gaussianDb(is,mo,So); % start with centered Gaussian prior 
outp  = @(w) compute_output('logistic_sigmoid',w(1:is),w(is+1),ds(:,1:is),'linear');

% Finding the posterior MAP value
max_iter = 10;
lr = 1;
w = zeros(is+1,1);
eps = 0.001;
for i=1:max_iter
   X = [ds(:,1:is),ones(size(ds,1),1)]; y = outp(w); t = ds(:,is+1);    % design - pred - labels 
   H = X'*diag(y.*(1-y))*X;                                             % full hessian
   g = So\(w-mo) -X'*(t-y);                                             % gradient
   d = (inv(So)+H)\g;
   w = w - lr*d;
   if (abs(lr*d)<eps)
       break;                                                           % stopping criterion 
   end
end

% Estimating the Hessian of log(posterior) at MAP value 
X = [ds(:,1:is),ones(size(ds,1),1)]; y = outp(w);
Sn = inv(inv(So) + X'*diag(y.*(1-y))*X);
wMap = w;


end
