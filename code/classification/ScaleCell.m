% Scaling design matrix: cell (m x 1) of observations (p x n)

function [X_] = ScaleCell(X)
    observations_nums = cellfun(@(m)( size(m, 1) ), X);
    num_features = size(X{1}, 2);

    X_mat = cell2mat(X);

    X_mat = Scale(X_mat);

    X_ = mat2cell(X_mat, observations_nums, num_features);
end
