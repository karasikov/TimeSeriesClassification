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
    @(tses)( mean(sqrt(sum(tses(1:3,:).^2, 1))) ),...
};

dataset = load_USCHAD_dataset();

segment_length = 200 : 200 : 2000;
mean_accuracy = zeros(size(segment_length));

for segment_length_idx = 1 : length(segment_length)

    [X,y] = GenerateFeatures(dataset, single_features, multi_features, ...
                                @(tses)( partition(tses, segment_length(segment_length_idx)) ), ...
                                @(fragments_features)( fragments_features ));

    %% Multi-class classification settings
    Parameters.coding = 'OneVsOne';
    Parameters.decoding = 'HD'; % 'LLB', 'ELB', 'ED', 'LAP', 'BDEN', 'AED', 'LLW', 'ELW'
    Parameters.base = 'SVM';
    Parameters.base_test = 'SVMtest';

    % Parameters.base_params.settings='-t 2'; % set SVM kernel
    % MulticlassSVMTuning(X, y, Parameters, {'c', 35:5:60; 'g', 0.0005:0.0005:0.004});

    Parameters.base_params.settings='-t 2 -c 45 -g 0.0025'; %for SVM binary classifier

    % X = Scale(X);
    % Parameters.base_params.settings='-t 2 -c 60 -g 0.05'; %for SVM binary classifier

    %% Analyze classification
    % Determine learn/test splitting parameters
    NSPLITS = 20;
    LEARN_RATE = 0.7;
    % Launch analyzer
    [confusion,sens] = AnalyseMulticlassClassification(X, y, ...
                                          @MulticlassClassificationTrain, Parameters, ...
                                          @MulticlassClassificationTest, ...
                                          LEARN_RATE, NSPLITS);
    disp(sens');

    mean_accuracy(segment_length_idx) = mean(sens);
end

plot(mean_accuracy)