load('../data/USC-HAD/USC_HAD_dataset.mat')

% ts_lengths = cellfun(@(sample)( size(sample.ts,1) ), dataset);
% mean_ts_length = mean(ts_lengths);

l = length(dataset);
ss = 512;
num_coeff = 50;
num_comps = 3;

X = zeros(l, (num_coeff * 2 + 1) * num_comps);
y = zeros(l, 1);

for i=1:length(dataset);
    y(i) = dataset{i}.class;

    for j=1:num_comps;
        ts = dataset{i}.ts(1:ss,j);
        ts = ts - mean(ts);
        X(i, (j-1) * (num_coeff * 2 + 1) + 1:j*(num_coeff * 2 + 1)) ...
            = model_fft((1:ss)', ts, num_coeff);
    end
end


% load dataset
[design_matrix, labels] = load_WISDM;
% replace NaN values with means

%labels(any(isnan(design_matrix), 2)) = [];
%design_matrix(any(isnan(design_matrix), 2),:) = [];

mean_features = mean(design_matrix(~any(isnan(design_matrix), 2),:));
for i = 1 : size(design_matrix, 1)
    design_matrix(i,isnan(design_matrix(i,:))) = mean_features(isnan(design_matrix(i,:)));
end

test_ind = randsample(1:size(design_matrix, 1), floor(size(design_matrix, 1) / 10));
train_ind = 1:size(design_matrix, 1);
train_ind(test_ind) = [];

% classification
addpath(genpath('classification'))
[accuracy,~,~,confusion]=MulticlassClassification(design_matrix(train_ind,:),...
                                                  labels(train_ind),...
                                                  design_matrix(test_ind,:),...
                                                  labels(test_ind));



% Mdl = arima(100,1,2)
% Mdl = estimate(Mdl, dataset{1}.ts(1:1000,4))
% y = Mdl.forecast(1000)
% plot([dataset{1}.ts(1:1000,4); y])

