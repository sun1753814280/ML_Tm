%% ========================================================================
% this code is to use all the samples for fitting final GRNN model
% Author: Zhangyu Sun
% Date: 2020/03/17
% Last Modified Date: 2021/02/05

function grnn_fitting(spread, train_data)

%% Prepare Data -----------------------------------------------------------

% set the input and target data
p_train = train_data(:,1:9);
t_train = train_data(:,10);
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

% construct the GRNN network
net = newgrnn(p, t, spread);
    
% save the model
save('model_grnn.mat','net','-v7.3');

% -------------------------- prediction -------------------------------
 
% normalize the test data
p_cv_test1 = mapminmax.apply(p_cv_test', p_set);
   
% prediction
l = length(p_cv_test1);
n = floor(l/1000);
test_out = zeros(1,l);

for j = 1:n
    disp(['compute predict value: ',num2str(j)]);
    test_out((j-1)*1000+1:j*1000) = sim(net, p_cv_test1(:,(j-1)*1000+1:j*1000));
end

test_out(n*1000+1:l) = sim(net, p_cv_test1(:,n*1000+1:l));

% normalization reverse
test_out = mapminmax.reverse(test_out, t_set);
        
% compute the prediction errors
pe = t_cv_test - test_out';
    
% obtain the output results
rsd = [p_cv_test inx_test pe t_cv_test test_out'];

% output the results
output_file = ['GRNN_',num2str(spread),'_Fit_results.mat'];
save(output_file,'rsd');

%% ----------------------------------------------------------------- END



