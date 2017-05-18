function [w,S] = gaussian_ep(ds, is, prior,plotflag)
% <============ HEADER =============>
% @brief    : computes the posterior approximation with Expectation
%             Progation approach (gaussian prior)
% @params   : ds <- dataset
%             is <- input vectors dimensionality
%             prior <- {mean,covariance matrix} structure -- assuming
%                       Gaussian prior for now 
%             [wL,SL] <- approximation given by the Laplace approximation
%             plotflag <- +1 for plot, 0 otherwise
% @returns  : posterior distribution q
% <============ HEADER =============>

% Initializes constants
n       = size(ds,1);   % number of points
m       = is+1;         % predictor dimensionality
targets = ds(:,m);      % targets 

% Initializes approximating distribution natural parameters (exponential family)
% r = (Sigma)^{-1}*mu
% beta = Sigma^{-1}
r       = zeros(m,n);               % (r1,r2,..,rn)
beta    = repmat(0.1*eye(m,m),1,1,n+1);  
beta(:,:,n+1) = prior.covmat;

% Gaussian EP loop 
outer_loop = 15;
r_var_array = zeros(outer_loop,1);
beta_var_array = zeros(outer_loop,1);
for outer_iter=1:outer_loop
    max_change_r = 0;
    max_change_beta = 0;
    for i=1:n
        x = [ds(i,1:is),1]';
        % Computes the natural parameters of the high-dimensional cavity
        beta_cavity = sum(beta,3)-beta(:,:,i);
        r_cavity = r* ones(n,1) - r(:,i);
        % Computes the mean and variance of the high-dimensional cavity
        Sigma_cavity = inv(beta_cavity);
        mu_cavity = beta_cavity\r_cavity;
        % Deduces mean and variance of the marginal
        v = x'*Sigma_cavity*x;
        mu = mu_cavity'*x;
        % Computes the mean and variance of the marginal hybrid (grid based) and
        % corresponding natural params
        [E,S] = hybrid_grid_based_approx(mu,v,targets(i));
        % Computes the natural parameters of the ith factor
        r_hybrid    = E/S - mu/v;
        beta_hybrid  = 1/S - (1/v);
        % Replaces in the approximating family
        max_change_r = max(max_change_r,norm(r_hybrid*x-r(:,i)));
        max_change_beta = max(max_change_beta,norm(beta_hybrid*(x*x')-beta(:,:,i)));
        r(:,i) = r_hybrid*x;
        beta(:,:,i) = beta_hybrid*(x*x');
    end
    r_var_array(outer_iter) = max_change_r;
    beta_var_array(outer_iter) = max_change_beta;
end
% Returning the approximation mean and variance (Gaussian approx)
S = inv(sum(beta,3));
w = S*(r* ones(n,1)+prior.mean);

% Plots
if (plotflag)
  figure; subplot(2,1,1);
  plot((1:outer_loop)',r_var_array,'LineWidth',2,'Color',[0.3 0.5 0.8]);
  xlabel('Iterations');
  ylabel('$\max_n \vert\vert \Delta r_n\vert\vert$','interpreter','latex','FontSize',14);
  legend('Maximum norm update');
  subplot(2,1,2);
  plot((1:outer_loop)',beta_var_array,'LineWidth',2,'Color',[0.8 0.5 0.3]);
  xlabel('Iterations');
  ylabel('$\max_n \vert\vert  \Delta \beta_n \vert\vert $','interpreter','latex','FontSize',14);
  legend('Maximum norm update');
end
end