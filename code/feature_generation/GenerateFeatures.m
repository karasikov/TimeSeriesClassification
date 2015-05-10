% Generate design matrix
%
% input: dataset --- [1 x dataset_size] struct array with fields:
%          ts    --- time series: [num_components x ts_length] double
%          label --- class label: integer in 1:num_classes
%
%        single_features --- set of function hundles for one-component time-series
%          (  single_features = { @(ts)( mean(ts) ), ... }  )
%
%        multi_features --- set of function hundles for multi-varied time-series
%          (  single_features = { @(tses)( mean(sqrt(sum(tses.^2, 1))) ), ... }  )
%
% output: X --- design matrix
%         y --- class label vector
%
function [X, y] = GenerateFeatures(dataset, single_features, multi_features)

%% class labels
y = [dataset.label]';

num_components = size(dataset(1).ts, 1);

%% design matrix creating
for i = 1 : length(dataset)
    feature_ind = 1;
    for j = 1 : length(single_features)
        for k = 1 : num_components
            features = feval(single_features{j}, dataset(i).ts(k,:));
            X(i, feature_ind : feature_ind + length(features) - 1) = features;
            feature_ind = feature_ind + length(features);
        end
    end
    for j = 1 : length(multi_features)
        features = feval(multi_features{j}, dataset(i).ts);
        X(i, feature_ind : feature_ind + length(features) - 1) = features;
        feature_ind = feature_ind + length(features);
    end
end

end