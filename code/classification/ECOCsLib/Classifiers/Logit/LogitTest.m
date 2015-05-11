%##########################################################################
% SVM prediction for (ECOClib Sergio Escalera)
%##########################################################################

function classes=SVMtest(data,classifier,params)

classes = glmval(classifier, data, 'logit');

classes = classes * 2 - 1;
