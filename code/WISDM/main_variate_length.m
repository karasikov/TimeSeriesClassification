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
    @(ts)( CalcArGarch(ts, 10, 2, 1) ),...
    @(ts)( CalcEigenvalues(ts, 10) ),...
    @(ts)( mean(ts) ),...
    @(ts)( std(ts) ),...
    @(ts)( mean(abs(ts - mean(ts))) ),...
    @(ts)( BinsCount(ts, 3) ),...
...
    % @(ts)( CalcAR(ts, 10) ),...
};

% tses: [num_components x ts_len] double
multi_features = {...
    @(tses)( mean(sqrt(sum(tses.^2, 1))) ),...
...
    % @(tses)( CalcAR(sqrt(sum(tses.^2, 1)), 10) ),...
};

Parameters = [];

dataset = load_WISDM_preprocessed_large();
dataset = dataset(randperm(length(dataset)));
dataset = dataset(1:500);

segm_size = round(96 ./ (1:4));
mean_accuracy = zeros(size(segm_size));

for segm_size_idx = 1 : length(segm_size)

    [X,y] = GenerateFeatures(dataset, single_features, multi_features, ...
                                @(tses)( partition(tses, segm_size(segm_size_idx)) ), ...
                                @(fragments_features)( fragments_features ));
    X = ScaleCell(X);

    %% Multi-class classification settings
    Parameters.coding = 'OneVsOne';
    Parameters.decoding = 'HD'; % 'LLB', 'ELB', 'ED', 'LAP', 'BDEN', 'AED', 'LLW', 'ELW'
    Parameters.base = 'SVM';
    Parameters.base_test = 'SVMtest';

    Parameters.base_params.settings='-t 2 -c 6 -g 0.21'; %for SVM binary classifier
    % MulticlassSVMTuning(X, y, Parameters, {'c', 8:0.5:12; 'g', 0.08:0.02:0.16});
    %Parameters.base_params.settings='-t 1 -d 3 -r 1 '; %for SVM binary classifier
    Parameters.base_params.settings='-t 0 -c 4';

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

    mean_accuracy(segm_size_idx) = mean(sens);

end

figure
plot(mean_accuracy)