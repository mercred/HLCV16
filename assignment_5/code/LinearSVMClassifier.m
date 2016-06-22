function LinearSVMClassifier(deCAF)
% libsvm should be installed before running this code.
% refer to the directory /libsvm-3.21/matlab'
% and
% run make
% addpath to the libsvm toolbox
addpath('libsvm-3.21/matlab');

% addpath to the miscellaneous
% addpath('/lbpSvm/use_libsvm');
addpath('use_libsvm');


%Prepare the features to be compatible with the data variable 
%Supplement Code Here
trainData = deCAF.train_data;
trainLabel = deCAF.train_labels;

testData = deCAF.test_data;
testLabel = deCAF.test_labels;

labelList = deCAF.label_list;

[trainData, trainData_mu, trainData_sigma] = featureNormalize(trainData);   %%% IMPORTANT

% Extract important information
NClass = length(labelList);
[Ntrain,D] = size(trainData);

[testData, ~, ~] = featureNormalize(testData, trainData_mu, trainData_sigma);  %%% IMPORTANT

% #######################
% Parameter selection
% #######################
% First we randomly pick some observations from the training set for parameter selection
tmp2 = randperm(Ntrain);
evalIndex = tmp2(1:Ntrain);
evalData = trainData(evalIndex,:);
evalLabel = trainLabel(evalIndex,:);

% #######################
% Automatic Cross Validation
% Parameter selection using n-fold cross validation
% #######################
optionCV.c = 1;
optionCV.stepSize = 5; 
optionCV.bestLog2c = 0;
optionCV.Nlimit = 2000;
Ncv_param = 5;% Ncv-fold cross validation cross validation

optionCV.epsilon = 0.005;

%% RBF Put the kernel Phi(data) rbf
disp('RBF')
optionCV.bestLog2g = log2(1/D);
optionCV.svmCmd = '-q -s 0 -t 2'; optionCV.gamma = 1/D; % -b 0 -v 5 -m 1000  -c 1

[bestc,bestg, bestcv] = automaticParameterSelection(evalLabel, evalData, Ncv_param, optionCV);

% rbf
cmd = [optionCV.svmCmd,' -b 1 -c ',num2str(bestc),' -g ',num2str(bestg)];
% Train the SVM
model = svmtrain(trainLabel, trainData, cmd);
% Use the SVM model to classify the data
[predictedLabel, tesacc, decisValueWinner] = svmpredict(testLabel, testData, model, '-b 1'); % run the SVM model on the test data
