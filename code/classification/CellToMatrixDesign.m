function [X_out, y_out, observation_indexes] = CellToMatrixDesign(X, y)
%
% Transform design matrix in cell format to standard matrix format
%

X_out = cell2mat(X);

y_out = [];
observation_indexes = [];
for i = 1 : length(X)
    num_observations = size(X{i}, 1);
    y_out = [y_out; repmat(y(i), num_observations, 1)];
    observation_indexes = [observation_indexes; repmat(i, num_observations, 1)];
end

end