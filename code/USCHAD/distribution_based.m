%% add path
if (length(mfilename()))
    cur_dir = fileparts(which(mfilename()));
else
    cur_dir = pwd;
end
addpath(genpath(strcat(cur_dir, '/../classification')));
addpath(genpath(strcat(cur_dir, '/../data_loading')));
addpath(genpath(strcat(cur_dir, '/../feature_generation')));
addpath(genpath(strcat(cur_dir, '/../segmentation')));

%% feature generation
% statistical features for one-component ts: [1 x ts_len] double
single_features = {...
    @(ts)( mean(ts) ),...
    @(ts)( ECDF_interpolated(ts, 8) ), ...
    @(ts)( std(ts) ),...
    @(ts)( mean(abs(ts - mean(ts))) ),...
...
    % @(ts)( CalcAR(ts, 10) ),...
};

% tses: [num_components x ts_len] double
multi_features = {...
    @(tses)( CalcArGarch(sqrt(sum(tses(1:3,:).^2, 1)), 10, 2, 1) ),...
    @(tses)( CalcArGarch(sqrt(sum(tses(4:6,:).^2, 1)), 10, 2, 1) ),...
    @(tses)( CalcEigenvalues(sqrt(sum(tses(1:3,:).^2, 1)), 10) ),...
    @(tses)( CalcEigenvalues(sqrt(sum(tses(4:6,:).^2, 1)), 10) ),...
    @(tses)( mean(sqrt(sum(tses(1:3,:).^2, 1))) ),...
    @(tses)( mean(sqrt(sum(tses(4:6,:).^2, 1))) ),...
...
    % @(tses)( CalcAR(sqrt(sum(tses.^2, 1)), 10) ),...
};

[X, y] = GenerateFeatures(load_USCHAD_dataset(), ...
                          single_features, multi_features, ...
                          @(tses)( partition(tses, 200) ), ...
                          @(fragments_features)( mean(fragments_features, 1) ));
X = Scale(X);

%% Multi-class classification settings
approaches  = {'OneVsAll', 'OneVsOne', 'ECOCRandom', 'ECOCBCH'}; %One-Vs-All, One-Vs-One, ECOC-Random, ECOC-BCH
classifiers = {'SVM', 'SVM', 'ADA', 'Logit'};
tester      = {'SVMtest', 'SVMMtest', 'ADAMtest', 'LogitTest'};
decoding    = {'HD', 'LLB', 'ELB', 'ED', 'LAP', 'BDEN', 'AED', 'LLW', 'ELW'};

clear Parameters;
Parameters.iterations=1000; %ECOC-Random parameter
Parameters.columns=18; %ECOC-Random parameter: code length
Parameters.BCHcodelength=15; %ECOC-BCH parameter: code length
Parameters.decoding='HD'; %Hamming distance
Parameters.base_params.iterations=50; %for AdaBoost binary classifier
Parameters.base_params.settings='-c 8 -t 1 -d 3 -r 1'; %for SVM binary classifier

k=2;
j=1;

Parameters.base=classifiers{k};
Parameters.base_test=tester{k};
Parameters.decoding=decoding{k};
Parameters.coding=approaches{j};

Parameters.base_params.settings='-t 2'; % set SVM kernel
% MulticlassSVMTuning(X, y, Parameters, {'c', 1:5:11; 'g', 0.005:0.05:0.95});

Parameters.base_params.settings='-t 1 -c 25 -r 1 -d 5'; %for SVM binary classifier --- 89%
Parameters.base_params.settings='-t 2 -c 20 -g 0.15'; %for SVM binary classifier --- 89.8%

%% Analyze classification
% Determine learn/test splitting parameters
NSPLITS = 20;
LEARN_RATE = 0.7;
% Launch analyzer
[~,sens] = AnalyseMulticlassClassification(X, y, ...
                                      @MulticlassClassificationTrain, Parameters, ...
                                      @MulticlassClassificationTest, ...
                                      LEARN_RATE, NSPLITS);
disp(sens');