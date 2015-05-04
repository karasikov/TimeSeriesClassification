%##########################################################################
% SVM probability prediction for (ECOClib Sergio Escalera)
%##########################################################################

function codeword = SVMPtest(data,classifier,params)

[labels,~,probs] = svmpredict(zeros(size(data,1),1), data, classifier, '-b 1 -q');

codeword = labels .* max(probs')';
