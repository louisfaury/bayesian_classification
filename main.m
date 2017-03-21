%% Bayesian computation main script
%  Breast Cancer Wiscousin Dataset 
%  Several GLM and Bayesian methods tests

close all;
clear all;
clc;
addpath(genpath('dataset'));
addpath(genpath('methods'));
addpath(genpath('utils'));

%% load dataset 
dataset_name = 'bcw';
ds = readtable(strcat(dataset_name,'.csv'));
[ds, is] = normalize_data(dataset_name,ds);

%% data visualization 
%visualize_data(ds, is);

% IRLS solution vizualisation 
[w, lc] = irls(ds, is); % TODO : add bias and penalization options
visualize_solution(w, ds, is, lc);

%% F-fold CV  %%TODO
[training_data, testing_data] = sample_train_test(ds, 0.9);
%irls_cv(w,training_data, test_data, input_size);
