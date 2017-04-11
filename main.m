%% Bayesian computation main script
%  Breast Cancer Wiscousin Dataset 
%  Several GLM and Bayesian methods tests

close all;
%clear all;
clc;
addpath(genpath('dataset'));
addpath(genpath('methods'));
addpath(genpath('utils'));

%% load dataset 
%dataset_name = 'bcw';
dataset_name = 'pidd';
ds = readtable(strcat(dataset_name,'.csv'));
[ds, is] = normalize_data(dataset_name,ds);


%% data visualization 
  % visualize_data(ds, is);


%% solution vizualisation
% IRLS solution vizualisation 
% ---------------------------
%  opt_up = struct('name','unpenalized','hp',[]);     % unpenalized IRLS
%  opt_L1 = struct('name','L1','hp',2);               % LASSO penalization
%  opt_L2 = struct('name','L2','hp',10);              % RIDGE penalization
%  opt = opt_L2;
 % [w, prior, lc] = irls(ds, is, opt); 
 % visualize_solution(w(1:is), ds, is, lc, opt);
% ---------------------------
% ---------------------------
% Bayesian learning and visualization
% - - - - - - - - - - - - - - - - - -
% laplace approximation for posterior
% - - - - - - - - - - - - - - - - - -
 [w,S] = laplax(ds,is);
 opt.laplax = struct('name','Laplace approximation');
 visualize_solution(w(1:is), ds, is, 1, opt.laplax);    % Basic visualization 
 %visualize_pdb(w,S,is)                                 % Visualization  TODO 
% - - - - - - - - - - - - - - - - - -
% grid based posterior 
% - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - -
% Expectation-propagation
% - - - - - - - - - - - - - - - - - -
    


%% F-fold CV  
fold = 10;
% irls cross-validation 
% ---------------------------
%irls_cv(ds, is, fold);
% bayesian cross-validation
% ---------------------------
% TODO : use predictive distribution for guess - see how it improves ! 


