%% ========================================================================
% this program is to select the optimal hyperparameters for different ML
% models using 10-fold cross validation
% Author: Zhangyu Sun
% Date: 2020/03/17
% Last Modified Date: 2021/02/05

close all; clc; clear all;

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

%% ------------------------------ RF method --------------------------------

disp('RF Method:...')

% set tree series
trees  = 5:10:95;

% loop for CV
for i = 1:length(trees)

    disp(i);
    
    % obtain the tree number
    tree = trees(i);
    
    % perform CV
    rf_cv(tree,train_data);
    
end

disp('done!');

%% -------------------------- BP method -----------------------------------

disp('BPNN Method:...');

% set the number of neurons in hidden layer
neurons = 7:19;

% loop for CV
for i = 1:length(neurons)

    disp(i);
    
    % get the neuron
    neuron = neurons(i);

    % perform CV
    bp_cv(neuron, train_data);
    
end

disp('done!');

%% -------------------------- GRNN method ----------------------------------

disp('GRNN Method:...');

% set the spread value series
spreads  = 0.01:0.01:0.1;

% loop for CV
for i = 1:length(spreads)

    disp(i);
    
    % get the spread value
    spread = spreads(i);
    
    % perform CV
    grnn_cv(spread, train_data);
    
end

disp('done!');

%% ------------------------------------------------------------------ END



