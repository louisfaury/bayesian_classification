function [wMap,Sn] = vb_normal(ds, is, prior)
% <============ HEADER =============>
% @brief    : computes the posterior approximation with Variational Bayes
%             approach (assuming location scale prior)
%             feature space is input space 
% @params   : ds <- dataset
%             is <- input vectors dimensionality
%             prior <- {mean,covariance matrix} structure -- assuming
%                       Gaussian prior for now 
% @returns  : posterior distribution q
% <============ HEADER =============>

