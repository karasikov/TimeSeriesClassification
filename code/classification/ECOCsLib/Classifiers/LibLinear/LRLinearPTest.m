%##########################################################################
% LibLinear LR prediction for (ECOClib Sergio Escalera)
%##########################################################################

function [codeword] = SVMLinearPTest(data, classifier, params)

[labels,~,probs] = liblinearpredict(zeros(size(data,1),1), sparse(data), classifier, '-b 1 -q');

codeword = labels .* max(probs')';

end