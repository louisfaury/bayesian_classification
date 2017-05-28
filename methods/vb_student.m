function [w,S] = vb_student(ds, is, prior,wL,SL,plotflag)
% <============ HEADER =============>
% @brief    : computes the posterior approximation with Variational Bayes
%             approach (assuming student prior)
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
n           = size(ds,1);           % dataset size
m           = is+1;                 % dataset dim
nu          = prior.nu;             % prior DoF
sig         = @(x) 1/(1+exp(-x));   % sig(.) (GLM) function
t           = ds(:,is+1);           % target vector 
eps         = 0.001;                % convergence criterion

 % computes log(posterior) (% up to a constant)
    function res = lpp(w)
        y   = compute_output('logistic_sigmoid',w(1:is),w(is+1),ds(:,1:is),'linear');
        res = -cross_entropy_loss_function(y,t) + sum(((nu+1)/2).*log(1+((w.^2)./nu)));
    end
% computes p'/p (d/dtheta{log p})
    function res = pP_o_p(x,w)
        y       = compute_output('logistic_sigmoid',w(1:is),w(is+1),x(:,1:is),'linear');
        res     = x'*(t-y) + diag((nu+1)./(nu+w.^2))*w;
    end

% Initialize the location scale Gaussian approximation distribution 
mu    = wL;
Sigma = sqrt(eigs(SL,1))*eye(m);

% Optimization loop
max_iter   = 200;
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
    Sigma = Sigma + 0.2*lr*(dSigma+inv(Sigma));
    lr = lr * 0.99;
end

% ELBO plot
if (plotflag)
    figure;
    plot(elbo(1:iter),'r-','LineWidth',2);
    legend('ELBO');
    title('ELBO stochastic maximization');
end

w = mu;
S = Sigma*Sigma';

end