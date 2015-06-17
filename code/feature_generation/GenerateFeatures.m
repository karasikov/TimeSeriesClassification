function [X, y] = GenerateFeatures(dataset, single_features, multi_features,...
                                                fragmentation, distr_params)
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
%          (  multi_features = { @(tses)( mean(sqrt(sum(tses.^2, 1))) ), ... }  )
%
%        fragmentation --- function handler for fragmenting time-series
%          (  fragmentation = @(tses)( {tses(:, 1 : length(tses) / 2),
%                                       tses(:, length(tses) / 2 : end)} )  ) %bisect
%
%        distr_params --- function handler for getting features' distribution
%                           parameters of time-series' fragments
%                         This parameters work as final features of initial ts
%          (  distr_params = @(fragments_features)( mean(fragments_features, 1) )  ) 
%
% output: X --- (dataset_size x 1) cell of (p x n) matrices, p --- number of observations per object,
%                                                            n --- number of features per observation
%         y --- vector of class labels
%

%% default distribution functions --- nothing to be done, just use features from ts

if nargin < 5
    fragmentation = @(tses)( {tses} );
    distr_params = @(fragments_params)( mean(fragments_params, 1) );
end

%% class labels
y = [dataset.label]';

num_components = size(dataset(1).ts, 1);

%% design matrix creating
for i = 1 : length(dataset)
    fragments = fragmentation(dataset(i).ts); % { first_fragment,
                                              %   second_fragment,
                                              %   ... }

    fragments_features = []; % [ first_fragment_features;
                             %   second_fragment_features;
                             %   ... ]

    for f = 1 : length(fragments)
        fragment = fragments{f};
        % each multi-varied time-series' fragment produces features 
        % |fragment_features| --- vector of fixed length

        fragment_features = [];

        for j = 1 : length(single_features)
            for k = 1 : num_components
                features = feval(single_features{j}, fragment(k,:));
                fragment_features = [fragment_features, features(:)'];
            end
        end
        for j = 1 : length(multi_features)
            features = feval(multi_features{j}, fragment);
            fragment_features = [fragment_features, features(:)'];
        end
        fragments_features(f,:) = fragment_features;
    end
    X{i,1} = distr_params(fragments_features);
end

end