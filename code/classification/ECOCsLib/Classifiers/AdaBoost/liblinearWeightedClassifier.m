%##########################################################################
% liblinear weak classifier
%##########################################################################

function WeakClassifier = liblinearWeightedClassifier(labels, data, weights)
    WeakClassifier = liblinearweighedtrain(weights, labels, sparse(data), '-c 1 -q');
