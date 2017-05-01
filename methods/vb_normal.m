function [w,S] = vb_normal(ds, is, prior,wL,SL,plotflag)
% <============ HEADER =============>
% @brief    : computes the posterior approximation with Variational Bayes
%             approach (assuming gaussian prior)
%             approximating distribution : location-scale Gaussian 
%             feature space is input space 
% @params   : ds <- dataset
%             is <- input vectors dimensionality
%             prior <- {mean,covariance matrix} structure -- assuming
%                       Gaussian prior for now 
%             plotflag <- +1 for plot, 0 otherwise
% @returns  : posterior distribution q
% <============ HEADER =============>

% Constants 
n = size(ds,1);
m = is+1;
a = prior.mean;
B = prior.covmat;
sig = @(x) 1/(1+exp(-x));
t   = ds(:,is+1);
eps = 0.001;
    % computes log(posterior)
    function res = lpp(w)
        y   = compute_output('logistic_sigmoid',w(1:is),w(is+1),ds(:,1:is),'linear');
        res = -cross_entropy_loss_function(y,t) -0.5*m*log(2*pi*det(B)) - 0.5*(w-a)'*(B\(w-a));
    end
% computes p'/p (d/dtheta{log p})
    function res = pP_o_p(x,w)
        y       = compute_output('logistic_sigmoid',w(1:is),w(is+1),x(:,1:is),'linear');
        res     = (-B\(w-a) + x'*(t-y));
    end
% computes d^2/dtheta(log p)
    function res = pP2(x,w)
       y = compute_output('logistic_sigmoid',w(1:is),w(is+1),x(:,1:is),'linear');
       res = -x'*diag(y.*(1-y))*x - inv(B);
    end

% Initialize the location scale Gaussian approximation distribution 
mu    = wL;
Sigma = SL;

% Optimization loop
max_iter   = 10;
num_sample = 100;
phi = [ds(:,1:is),ones(n,1)];
lr = 0.01;
elbo = zeros(max_iter,1);
for iter = 1:max_iter
    samples = randn(m,num_sample);
    dMu = zeros(m,1);
    dSigma = zeros(m);
    for i=1:num_sample
       elbo(iter,1) = elbo(iter,1) - (1/num_sample) * (0.5*m*log(2*pi*det(Sigma*Sigma')) + 0.5*(samples(:,i)-mu)'*((Sigma*Sigma')\(samples(:,i)-mu))+ lpp(samples(:,i)));
       dMu = dMu + pP_o_p(phi,mu + Sigma*samples(:,i))/num_sample;
       dSigma = dSigma + (pP_o_p(phi,mu + Sigma*samples(:,i))*samples(:,i)')/num_sample;
    end
    mu = mu + lr*dMu;
    Sigma = Sigma + 0.1*lr*(dSigma+inv(Sigma));
    lr = lr * 0.99;
end

% ELBO plot
if (plotflag)
    figure;
    plot(elbo(1:iter),'r-','LineWidth',2);
    legend('ELBO');
    title('ELBO stochastic maximization');
end

%% TODO : control over ELBOW
w = mu;
S = Sigma*Sigma';


end