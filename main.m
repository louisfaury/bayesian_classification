%% Bayesian computation main script
%  PIMA Indian dataset 
%  Several GLM and Bayesian methods tests

close all;
clc;
addpath(genpath('dataset'));
addpath(genpath('methods'));
addpath(genpath('utils'));
addpath(genpath('lib/sparse_bayes'))
%% load dataset 
%dataset_name = 'bcw';
%dataset_name = 'piddr';
%dataset_name = 'synth';
dataset_name = 'park';
ds = readtable(strcat(dataset_name,'.csv'));
[ds, is] = normalize_data(dataset_name,ds);


%% data visualization 
% visualize_data(ds, is);


%% solution vizualisation

% IRLS solution vizualisation 
% ---------------------------
% opt_up = struct('name','unpenalized','hp',[]);     % unpenalized IRLS
% opt_L1 = struct('name','L1','hp',0.01);            % LASSO penalization
% opt_L2 = struct('name','L2','hp',1);               % RIDGE penalization
% opt = opt_L1;
% [w, prior, lc] = irls(ds, is, opt); 
% visualize_solution(w(1:is), ds, is, lc, opt);
% ---------------------------
%%TODO donut representation 

% ---------------------------
% Bayesian learning and visualization
% - - - - - - - - - - - - - - - - - -
% defining a Gaussian prior 
prior.mean = zeros(is+1,1);
prior.covmat = eye(is+1), % - 9*double([1:is+1]==11)'*double([1:is+1]==11); - 9*double([1:is+1]==14)'*double([1:is+1]==14);

% laplace approximation for posterior
% - - - - - - - - - - - - - - - - - -
% [wLg,SLg] = laplax_normal(ds,is,prior);
% visualize_pdb(ds,wLg,SLg,is)                  % Visualization  (predictive distribution)
% - - - - - - - - - - - - - - - - - -
% Variational Bayes 
% - - - - - - - - - - - - - - - - - -
% [w,S] = vb_normal(ds, is, prior, wLg, SLg, true);
% visualize_pdb(ds,w,S,is)
% - - - - - - - - - - - - - - - - - -
% Expectation-propagation
% - - - - - - - - - - - - - - - - - -
% [wEp,SEp] = gaussian_ep(ds, is, prior,1);    
% visualize_pdb(ds,wEp,SEp,is)

% defining a Student prior 
% prior.nu = 2*ones(is+1,1);
% laplace approximation for posterior
% - - - - - - - - - - - - - - - - - -
% [wLs,SLs] = laplax_student(ds,is,prior,true);
% visualize_pdb(ds,wLs,SLs,is)                  % Visualization  (predictive distribution)
% - - - - - - - - - - - - - - - - - -
% Variational Bayes 
% - - - - - - - - - - - - - - - - - -
% [w,S] = vb_student(ds, is, prior, wLs, SLs, true);
% visualize_pdb(ds,w,S,is)


% - - - - - - - - - - - - - - - - - -
% Non-linear method (RVR)
% - - - - - - - - - - - - - - - - - -
width = 0.5;
[w,y] = rvm_train(ds,is,width);
rvm_visualize(ds,is,width);

%% F-fold CV  
fold = 100;
% irls cross-validation 
% ---------------------------
irls_cv(ds, is, fold);
% bayesian cross-validation
% ---------------------------


