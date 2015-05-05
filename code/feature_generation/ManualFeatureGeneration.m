function [X, y] = ManualFeatureGeneration(dataset)

%% class labels
y = [dataset.label]';

%% manual features

num_bins = 10; % for histogram
function bins = bins_count(ts)
    edges = linspace(min(ts), max(ts), num_bins + 1);
    bins = histc(ts, edges) / length(ts);
    bins(end - 1) = bins(end - 1) + bins(end);
    bins(end) = [];
end

% statistical features for one-component ts: [1 x ts_len] double
single_features = {...
    @(ts)( bins_count(ts) ),...
    @(ts)( mean(ts) ),...
    @(ts)( std(ts) ),...
    @(ts)( mean(abs(ts - mean(ts))) ),...
};

% tses: [num_components x ts_len] double
multi_features = {...
    @(tses)( mean(sqrt(sum(tses.^2, 1))) ),...
};

num_components = size(dataset(1).ts, 1);

% Initialize design matrix
X = zeros(length(dataset), num_bins * num_components + ...
                           length(single_features) + ...
                           length(multi_features) * num_components);

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