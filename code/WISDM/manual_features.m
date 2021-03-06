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
    @(ts)( mean(ts) ),...
    @(ts)( std(ts) ),...
    @(ts)( mean(abs(ts - mean(ts))) ),...
    @(ts)( BinsCount(ts, 10) ),...
};

% tses: [num_components x ts_len] double
multi_features = {...
    @(tses)( mean(sqrt(sum(tses.^2, 1))) ),...
};

[X, y] = GenerateFeatures(load_WISDM_preprocessed_large(), ...
                          single_features, multi_features);

%% Multi-class classification settings
Parameters.coding = 'OneVsOne';
Parameters.decoding = 'HD';
Parameters.base = 'SVM';
Parameters.base_test = 'SVMtest';

% MulticlassSVMTuning(X, y, Parameters, {'c', 6:2:12; 'g', 0.03 * 2.^(2:5)});

Parameters.base_params.settings='-t 2 -c 8.5 -g 0.12'; %for SVM binary classifier

%% Analyze classification
% Determine train/test splitting parameters
NSPLITS = 50;
LEARN_RATE = 0.7;
% Launch analyzer
accuracy_figure_name = ['Accuracy_Dataset_WISDM_manual_features' ...
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