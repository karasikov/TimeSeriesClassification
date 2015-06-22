%##########################################################################
% SVM liblinear prediction for (ECOClib Sergio Escalera)
%##########################################################################

function [classes] = SVMLinearMTest(data, classifier, params)

[~,~,classes] = liblinearpredict(zeros(size(data,1),1), sparse(data), classifier, '-q');

end