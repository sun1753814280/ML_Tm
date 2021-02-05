%% ========================================================================
% this code is to 10-fold CV to select the optimal neuron number for bp
% Author: Zhangyu Sun
% Date: 2020/03/18
% Last Modified Date: 2021/02/05

function bp_cv(neuron, train_data)

%% Prepare Data -----------------------------------------------------------

% set the input and target data
p_train  = train_data(:,1:9);
t_train  = train_data(:,10);
site_inx = train_data(:,11);

%% K-Fold Cross Validation ------------------------------------------------

% set k
k = 10;

% set the indices for K-fold CV
indices = crossvalind('Kfold', length(t_train), k); % 5-fold CV

% initialize residual
rsd = [];

% Loop for K-fold CV
for i = 1:k

    % print k-fold times
    disp(['k-fold times: ',num2str(i)]);
    
    % get the indeces of K-fold CV
    test_id = (indices == i); train_id =~ test_id;
    
    % get the train and test data for K-fold CV
    p_cv_train = p_train(train_id,:);
    t_cv_train = t_train(train_id,:);
    p_cv_test  = p_train(test_id,:);
    t_cv_test  = t_train(test_id,:);
    inx_test   = site_inx(test_id,:);
    
    % ------------------------ construct model ----------------------------
    
    % normalization
    [p, p_set] = mapminmax(p_cv_train');
    [t, t_set] = mapminmax(t_cv_train');

    % create the network
    net_bp = newff(p,t,neuron,{ 'tansig' 'purelin' } , 'trainlm');
    
    % set parameter
    net_bp.trainParam.epochs = 2000;

    % train the network
    net_bp = train(net_bp, p ,t);
    
    % -------------------------- prediction -------------------------------
 
    % normalize the test data
    p_cv_test1 = mapminmax.apply(p_cv_test', p_set);
   
    % prediction
    test_out = sim(net_bp, p_cv_test1);

    % normalization reverse
    test_out = mapminmax.reverse(test_out, t_set);
        
    % compute the prediction errors
    pe = t_cv_test - test_out';
    
    % obtain the output results
    results = [p_cv_test inx_test pe t_cv_test test_out'];
    
    % store the residuals
    rsd = [rsd;results];
    
end

% output the results
output_file = ['BPNN_',num2str(neuron),'_CV_results.mat'];
save(output_file,'rsd');

%% ----------------------------------------------------------------- END



