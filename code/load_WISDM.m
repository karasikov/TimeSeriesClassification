function [design_matrix, labels] = load_WISDM()

addpath('../data');
dataset = GetDatasetWISDM;

design_matrix = dataset.design_matrix;
labels = dataset.labels;

end