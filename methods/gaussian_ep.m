function [w,S] = gaussian_ep(ds, is, prior,wL,SL,plotflag)
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

end