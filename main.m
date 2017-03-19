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

%% data normalization and visualization 
[ds, input_size] = normalize_data(dataset_name,ds);
%visualize_data(ds);

%% train-test sub-datasets sample
[training_data, testing_data] = sample_train_test(ds, 0.9);

%% IRLS methods 
w = irls(training_data, input_size);
%irls_cv(w,training_data, test_data, input_size);

% TODO : add bias, perform CV

%% Cross validation 
