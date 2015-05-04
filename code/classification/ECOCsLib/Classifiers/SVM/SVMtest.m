%##########################################################################
% SVM prediction for (ECOClib Sergio Escalera)
%##########################################################################

function classes=SVMtest(data,classifier,params)

classes = svmpredict(zeros(size(data,1),1), data, classifier, '-q');
