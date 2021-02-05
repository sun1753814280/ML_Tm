%% ========================================================================
% this code is to use all the samples for fitting final RF model
% Author: Zhangyu Sun
% Date: 2020/03/18
% Last Modified Date: 2021/02/05

function rf_fitting(tree, train_data)

%% Prepare Data -----------------------------------------------------------

% set the input and target data
p_train  = train_data(:,1:9);
t_train  = train_data(:,10);
site_inx = train_data(:,11);

% get the train and test data for K-fold CV
p_cv_train = p_train;
t_cv_train = t_train;
p_cv_test  = p_train;
t_cv_test  = t_train;
inx_test   = site_inx;
    
% ------------------------ construct model ----------------------------
    
% normalization
[p, p_set] = mapminmax(p_cv_train');
[t, t_set] = mapminmax(t_cv_train');
p = p';
t = t';

% create and train the RF model
model_rf = TreeBagger(tree, p, t, 'Method', 'regression', 'OOBPrediction', 'on');  

% save the model
save('model_rf.mat','model_rf','-v7.3');

% -------------------------- prediction -------------------------------
 
% normalize the test data
p_cv_test1 = mapminmax.apply(p_cv_test', p_set);
p_cv_test1 = p_cv_test1';
   
% predict the results using trained network
test_out = predict(model_rf, p_cv_test1);

% normalization reverse
test_out = mapminmax.reverse(test_out, t_set);
        
% compute the prediction errors
pe = t_cv_test - test_out;
    
% obtain the output results
rsd = [p_cv_test inx_test pe t_cv_test test_out];

% output the results
output_file = ['RF_',num2str(tree),'_Fit_results.mat'];
save(output_file,'rsd');

%% ----------------------------------------------------------------- END



