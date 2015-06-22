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
    @(ts)( CalcAR(ts, 6) ),...
    @(ts)( mean(ts) ),...
    @(ts)( std(ts) ),...
    @(ts)( mean(abs(ts - mean(ts))) ),...
};

% tses: [num_components x ts_len] double
multi_features = {...
    @(tses)( mean(sqrt(sum(tses.^2, 1))) ),...
};

clear Parameters;

dataset = load_WISDM_preprocessed_large();
% dataset = dataset(randperm(length(dataset)));
% dataset = dataset(1:500);

num_segments = 1:5;
segm_size = round(length(dataset(1).ts(1,:)) ./ num_segments);
mean_accuracy = zeros(size(segm_size));

for segm_size_idx = 1 : length(segm_size)
    [X,y] = GenerateFeatures(dataset, single_features, multi_features, ...
                             @(tses)( partition(tses, segm_size(segm_size_idx)) ), ...
                             @(fragments_features)( fragments_features ));
    X = ScaleCell(X);

    %% Multi-class classification settings
    Parameters.coding = 'OneVsAll';
    Parameters.decoding = 'LLB';
    Parameters.base = 'SVMLinear';
    Parameters.base_test = 'SVMLinearMTest';

    % MulticlassSVMTuning(X, y, Parameters, {'c', 3:2:7});
    Parameters.base_params.settings='-c 60'; %for SVM binary classifier

    %% Analyze classification
    % Determine train/test splitting parameters
    NSPLITS = 5;
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