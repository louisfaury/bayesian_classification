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
 opt_up    = struct('name','unpenalized','hp',[]);  % unpenalized IRLS
 opt_L1 = struct('name','L1','hp',2);               % LASSO penalization
 opt_L2 = struct('name','L2','hp',10);              % RIDGE penalization
 opt = opt_L2;
 % [w, prior, lc] = irls(ds, is, opt); 
 % visualize_solution(w(1:is), ds, is, lc, opt);
 % bayesian learning 
 laplax(ds,is);


%% F-fold CV  
%fold = 10;
%irls_cv(ds, is, fold);
