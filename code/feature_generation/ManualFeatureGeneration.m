function [X, y] = ManualFeatureGeneration(dataset)

%% class labels
y = [dataset.label]';

%% manual features

% statistical features for one-component ts: [1 x ts_len] double
single_features = {...
    @(ts)( BinsCount(ts, 3) ),...
    @(ts)( mean(ts) ),...
    @(ts)( std(ts) ),...
    @(ts)( mean(abs(ts - mean(ts))) ),...
...
    @(ts)( CalcEigenvalues(ts, 10) ),...
    @(ts)( CalcArGarch(ts, 10, 2, 1) ),...
    % @(ts)( CalcAutoregression(ts, 10) ),...
};

% tses: [num_components x ts_len] double
multi_features = {...
    @(tses)( mean(sqrt(sum(tses.^2, 1))) ),...
...
    % @(tses)( CalcAutoregression(sqrt(sum(tses.^2, 1)), 10) ),...
};

num_components = size(dataset(1).ts, 1);

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