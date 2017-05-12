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
targets     = ds(:,is+1);           % target vector 
eps         = 0.001;                % convergence criterion