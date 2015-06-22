%% add path
if (length(mfilename()))
    cur_dir = fileparts(which(mfilename()));
else
    cur_dir = pwd;
end
addpath(genpath(strcat(cur_dir, '/../classification')));
addpath(genpath(strcat(cur_dir, '/../data_loading')));
addpath(genpath(strcat(cur_dir, '/../feature_generation')));

%% feature generation
% statistical features for one-component ts: [1 x ts_len] double
single_features = {...
    @(ts)( CalcAR(ts, 6) ),...
    @(ts)( mean(ts) ),...
    @(ts)( std(ts) ),...
    @(ts)( mean(abs(ts - mean(ts))) ),...
    % @(ts)( BinsCount(ts, 3) ),...
    % @(ts)( CalcEigenvalues(ts, 10) ),...
...
    % @(ts)( CalcAR(ts, 10) ),...
};

% tses: [num_components x ts_len] double
multi_features = {...
    @(tses)( mean(sqrt(sum(tses.^2, 1))) ),...
    % @(tses)( regress(tses(3,:)', tses(1:2,:)') ), ...
...
    % @(tses)( CalcAR(sqrt(sum(tses.^2, 1)), 10) ),...
};

[X, y] = GenerateFeatures(load_WISDM_preprocessed_large(), ...
                          single_features, multi_features);

X = ScaleCell(X);

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

% Parameters.coding='CUSTOM';
% Parameters.custom_coding=@(num_classes, params)( CUSTOM_ECOC() );
% Parameters.custom_coding_params = [];

% Parameters.base_params.settings='-t 2'; % set SVM kernel
% MulticlassSVMTuning(X, y, Parameters, {'c', 3:2:7; 'g', 0.15:0.01:0.25});

% Parameters.base_params.settings='-t 0 -c 10'; %for SVM binary classifier
Parameters.base_params.settings='-t 2 -c 4 -g 0.8'; %for SVM binary classifier
% Parameters.coding=approaches{2};
% Parameters.base_params.settings='-t 2 -c 6 -g 0.09'; %for SVM binary classifier


%% Analyze classification
% Determine learn/test splitting parameters
NSPLITS = 50;
LEARN_RATE = 0.7;
% Launch analyzer
accuracy_figure_name = ['Accuracy_Dataset_WISDM' ...
                        '_nSplits_' num2str(NSPLITS) ...
                        '_rate_' num2str(LEARN_RATE) ...
                        '_approach_' Parameters.coding ...
                        '_' Parameters.decoding ...
                        '_classifier_' Parameters.base ...
                        '_' Parameters.base_params.settings];

[confusion,sens] = AnalyseMulticlassClassification(X, y, ...
                                      @MulticlassClassificationTrain, Parameters, ...
                                      @MulticlassClassificationTest, ...
                                      LEARN_RATE, NSPLITS, ...
                                      regexprep(accuracy_figure_name, ' ', ''));
disp(sens');