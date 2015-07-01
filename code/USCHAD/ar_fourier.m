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
    @(ts)( CalcAR(ts, 10) ),...
};

% tses: [num_components x ts_len] double
multi_features = {...
    @(tses)( CalcAR(sqrt(sum(tses(1:3,:).^2, 1)), 10) ),...
    @(tses)( CalcAR(sqrt(sum(tses(4:6,:).^2, 1)), 10) ),...
    @(tses)( DFTCoefficients(sqrt(sum(tses(1:3,:).^2, 1)), 5) ),...
    @(tses)( DFTCoefficients(sqrt(sum(tses(4:6,:).^2, 1)), 5) ),...
};

% smoothing: round each 10 measurements
dataset = aggregate(load_USCHAD_dataset(), 10);

[X, y] = GenerateFeatures(dataset, single_features, multi_features);
X = ScaleCell(X);

%% Multi-class classification settings
approaches  = {'OneVsAll', 'OneVsOne', 'ECOCRandom', 'ECOCBCH'}; %One-Vs-All, One-Vs-One, ECOC-Random, ECOC-BCH
classifiers = {'SVM', 'SVM', 'ADA', 'Logit'};
tester      = {'SVMtest', 'SVMMtest', 'ADAMtest', 'LogitTest'};
decoding    = {'HD', 'LLB', 'ELB', 'ED', 'LAP', 'BDEN', 'AED', 'LLW', 'ELW'};

clear Parameters;

k=1;
j=2;

Parameters.base=classifiers{k};
Parameters.base_test=tester{k};
Parameters.decoding=decoding{k};
Parameters.coding=approaches{j};

% MulticlassSVMTuning(X, y, Parameters, {'c', 1:5:41; 'g', 0.01:0.03:0.2});

Parameters.base_params.settings='-t 2 -c 16 -g 0.04'; %for SVM binary classifier

%% Analyze classification
% Determine train/test splitting parameters
NSPLITS = 100;
LEARN_RATE = 0.7;
% Launch analyzer
accuracy_figure_name = ['Accuracy_Dataset_USCHAD_fourier_AR' ...
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