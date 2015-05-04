%##########################################################################
% SVM margin prediction for (ECOClib Sergio Escalera)
%##########################################################################

function classes=SVMMtest(data,classifier,params)

[~,~,classes] = svmpredict(zeros(size(data,1),1), data, classifier, '-q');
