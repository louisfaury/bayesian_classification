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
ds = dataset('File','breast-cancer-wisconsin.csv');
% TODO : pick training and testing dataset

%% IRLS methods 


%% Cross validaiton 
