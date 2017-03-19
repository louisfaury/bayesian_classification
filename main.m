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
dataset = 'bcw';
ds = readtable(strcat(dataset,'.csv'));
[training_data, testing_data, input_size] = sample_train_test(dataset,ds, 0.9);

%% IRLS methods 
w = irls(training_data, input_size);
%irls_cv(w,training_data, test_data, input_size);

% TODO : add bias, perform CV

%% Cross validaiton 
