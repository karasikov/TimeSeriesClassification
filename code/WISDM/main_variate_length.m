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

Parameters = [];

dataset_ = load_WISDM_preprocessed_large();

[X,y] = GenerateFeatures(dataset_, single_features, multi_features);

num_segments = 1:5;
mean_accuracy = zeros(size(num_segments));

for num_segments_idx = 1 : length(num_segments)

    NUM_SEGMENTS = num_segments(num_segments_idx);

    Parameters.observation_fragments_indexes = [];
    dataset = struct();
    for i = 1 : length(dataset_)
        ts = dataset_(i).ts;
        label = dataset_(i).label;
        Parameters.observation_fragments_indexes{i} = ...
            [(i - 1) * NUM_SEGMENTS + 1 : i * NUM_SEGMENTS];

        for j = 1 : NUM_SEGMENTS
            segment_length = floor(length(ts) / NUM_SEGMENTS);
            dataset((i - 1) * NUM_SEGMENTS + j).ts = ...
                ts(:,(j-1) * segment_length + 1 : j * segment_length);
            dataset((i - 1) * NUM_SEGMENTS + j).label = label;
        end
    end

    [X,~] = GenerateFeatures(dataset, single_features, multi_features);

    %% Multi-class classification settings
    Parameters.coding = 'OneVsOne';
    Parameters.decoding = 'HD'; % 'LLB', 'ELB', 'ED', 'LAP', 'BDEN', 'AED', 'LLW', 'ELW'
    Parameters.base = 'SVM';
    Parameters.base_test = 'SVMtest';

    Parameters.base_params.settings='-t 2'; % set SVM kernel
    % MulticlassSVMTuning(X, y, Parameters, {'c', 8:0.5:12; 'g', 0.08:0.02:0.16});

    Parameters.base_params.settings='-t 2 -c 8.5 -g 0.12'; %for SVM binary classifier
    %Parameters.base_params.settings='-t 1 -d 3 -r 1 '; %for SVM binary classifier

    %% Analyze classification
    % Determine learn/test splitting parameters
    NSPLITS = 5;
    LEARN_RATE = 0.7;
    % Launch analyzer
    [~,sens] = AnalyseMultiClassification(X, y, ...
                                          @MulticlassClassificationTrain, Parameters, ...
                                          @MulticlassClassificationTest, ...
                                          LEARN_RATE, NSPLITS);
    disp(sens');

    mean_accuracy(num_segments_idx) = mean(sens);

end