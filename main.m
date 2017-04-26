%% Bayesian computation main script
%  PIMA Indian dataset 
%  Several GLM and Bayesian methods tests

close all;
clc;
addpath(genpath('dataset'));
addpath(genpath('methods'));
addpath(genpath('utils'));

%% load dataset 
%dataset_name = 'bcw';
%dataset_name = 'piddr';
%dataset_name = 'synth';
dataset_name = 'park';
ds = readtable(strcat(dataset_name,'.csv'));
[ds, is] = normalize_data(dataset_name,ds);


%% data visualization 
visualize_data(ds, is);


%% solution vizualisation
% IRLS solution vizualisation 
% ---------------------------
% opt_up = struct('name','unpenalized','hp',[]);     % unpenalized IRLS
% opt_L1 = struct('name','L1','hp',0.01);               % LASSO penalization
% opt_L2 = struct('name','L2','hp',1);              % RIDGE penalization
% opt = opt_L1;
% [w, prior, lc] = irls(ds, is, opt); 
% visualize_solution(w(1:is), ds, is, lc, opt);
% ---------------------------
% ---------------------------
% Bayesian learning and visualization
% - - - - - - - - - - - - - - - - - -
% laplace approximation for posterior
% - - - - - - - - - - - - - - - - - -
% laplax_prior.mean = zeros(is+1,1);
% laplax_prior.covmat = 10*eye(is+1);%+ 10*double([1:is+1]==11)'*double([1:is+1]==11) + 10*double([1:is+1]==14)'*double([1:is+1]==14);
% [w,S] = laplax_normal(ds,is,laplax_prior);
% visualize_pdb(ds,w,S,is)                  % Visualization  (predictive distribution)
% - - - - - - - - - - - - - - - - - -
% grid based posterior 
% - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - -
% Expectation-propagation
% - - - - - - - - - - - - - - - - - -
    


%% F-fold CV  
fold = 200;
% irls cross-validation 
% ---------------------------
irls_cv(ds, is, fold);
% bayesian cross-validation
% ---------------------------


