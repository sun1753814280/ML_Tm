%% ========================================================================
% this program is to use all the samples for model fitting to yield the
% final model for prediction
% Author: Zhangyu Sun
% Date: 2020/03/17
% Last Modified Date: 2021/02/05

close all; clc; clear all;

%% -----------------------------------------------------------------------

% load the data
load('train_data.mat');
% Data content ------------------------------------------------------------
% Column 1  : latitude (degree)
% Column 2  : longitude (degree)
% Column 3  : height (m)
% Column 4  : year
% Column 5  : day of year (doy)
% Column 6  : hour of day (hod)
% Column 7  : surface temperature Ts (K)
% Column 8  : surface water vapor pressure es (hPa)
% Column 9  : weighted mean temperature from GPT3 model Tm_GPT3 (K)
% Column 10 : weighted mean temperature derived from radiosonde observations Tm (K)
% Column 11 : index of different radiosonde sites (1~150)
% -------------------------------------------------------------------------

% set the hyperparameters for different ML models
tree = 55;
neuron = 19;
spread = 0.06;

% perform modelg fitting for RF model
disp('RF Method:...')
rf_fitting(tree,train_data);
disp('done!');

% perform model fitting for BPNN model
disp('BPNN Method:...')
bp_fitting(neuron, train_data);
disp('done!');

% perform model fitting for GRNN model
disp('GRNN Method:...')
grnn_fitting(spread, train_data);
disp('done!');

%% ------------------------------------------------------------------ END



