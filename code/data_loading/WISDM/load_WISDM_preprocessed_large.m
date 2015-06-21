function [dataset] = load_WIDSM_preprocessed_large()
%
% ( 1 - Jogging,  2 - Walking,
%   3 - Upstairs, 4 - Downstairs,
%   5 - Sitting,  6 - Standing )
%
if (length(mfilename()))
    cur_dir = fileparts(which(mfilename()));
else
    cur_dir = pwd;
end
path_to_data = strcat(cur_dir, '/../../../data/WISDM/');

try
    load(strcat(path_to_data, 'preprocessed_large.mat'));
catch

    unzip(strcat(path_to_data, 'preprocessed_large.zip'), path_to_data);

    dat = csvread(strcat(path_to_data, 'preprocessed_large.csv'));
    labels = dat(:,1);
    dat = dat(:,2:end);

    dataset = struct([]);

    for i = 1 : size(dat, 1)
        % Divide data on X,Y,Z
        dataset(i).ts = zeros(3, 200);
        dataset(i).ts(1,:) = dat(i, 1 : 200);
        dataset(i).ts(2,:) = dat(i, 201 : 400);
        dataset(i).ts(3,:) = dat(i, 401 : 600);
        dataset(i).label = labels(i);
    end

    save(strcat(path_to_data, 'preprocessed_large.mat'), 'dataset');

    end

end