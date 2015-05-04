%##########################################################################
% liblinear predictor
%##########################################################################

function predicted_labels = liblinearWeightedPredictor(liblinearWeightedClassifier, data)
    [~,~,predicted_labels] = liblinearpredict(zeros(size(data,1),1), sparse(data), liblinearWeightedClassifier, '-q');
