if (length(mfilename()))
    cur_dir = fileparts(which(mfilename()));
else
    cur_dir = pwd;
end
addpath(genpath(strcat(cur_dir, '/../classification')));
addpath(genpath(strcat(cur_dir, '/../data_loading')));
addpath(genpath(strcat(cur_dir, '/../feature_generation')));

[X, y] = ManualFeatureGeneration(load_WISDM_preprocessed_large());

%% Multi-class classification settings
approaches  = {'OneVsAll', 'OneVsOne', 'ECOCRandom', 'ECOCBCH'}; %One-Vs-All, One-Vs-One, ECOC-Random, ECOC-BCH
classifiers = {'SVM', 'SVM', 'ADA'};
tester      = {'SVMtest', 'SVMMtest', 'ADAMtest'};
decoding    = {'HD', 'LLB', 'ELB', 'ED', 'LAP', 'BDEN', 'AED', 'LLW', 'ELW'};

clear Parameters;
Parameters.iterations=1000; %ECOC-Random parameter
Parameters.columns=18; %ECOC-Random parameter: code length
Parameters.BCHcodelength=15; %ECOC-BCH parameter: code length
Parameters.decoding='HD'; %Hamming distance
Parameters.base_params.iterations=50; %for AdaBoost binary classifier
Parameters.base_params.settings='-c 8 -t 1 -d 3 -r 1'; %for SVM binary classifier

k=1;
j=2;

Parameters.base=classifiers{k};
Parameters.base_test=tester{k};
Parameters.decoding=decoding{k};
Parameters.coding=approaches{j};

%% Analyze classification
addpath('../../../../ErrorAnalysis');
% Determine learn/test splitting parameters
NSPLITS = 50;
LEARN_RATE = 0.7;
% Launch analyzer
err_test = AnalyseMultiClassification(X, y, ...
                                      @MulticlassClassificationTrain, Parameters, ...
                                      @MulticlassClassificationTest, ...
                                      LEARN_RATE, NSPLITS);

err_test