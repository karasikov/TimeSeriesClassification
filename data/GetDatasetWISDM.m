% WISDM_dataset

function dataset = GetDatasetWISDM

if ~exist('..\data\WISDM\WISDM_transformed.mat', 'file')
    if ~exist('..\data\WISDM\WISDM_ar_v1.1\WISDM_ar_v1.1_transformed.arff', 'file')
        untar('http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz', '..\data\WISDM')
    end

    M = importdata('..\data\WISDM\WISDM_ar_v1.1\WISDM_ar_v1.1_transformed.arff');
    first_line = find(~cellfun('isempty', strfind(M, '@data'))) + 1;

    classes = {'Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing'};
    labels = {1, 2, 3, 4, 5, 6};
    label = containers.Map(classes, labels);

    dataset = {};
    design_matrix = zeros(length(M) - first_line, 43); % 43 features in transformed data
    labels = zeros(length(M) - first_line, 1);

    for i = first_line : length(M)
        record = M{i};
        tokens = strsplit(record, ',');
        design_matrix(i - first_line + 1,:) = str2double(tokens(3:45));
        labels(i - first_line + 1) = label(tokens{46});
    end

    dataset.design_matrix = design_matrix;
    dataset.labels = labels;
    save('..\data\WISDM\WISDM_transformed.mat', 'dataset');
end

load('..\data\WISDM\WISDM_transformed.mat', 'dataset');

end