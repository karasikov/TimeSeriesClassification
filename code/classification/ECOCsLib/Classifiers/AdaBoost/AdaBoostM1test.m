%##########################################################################
% AdaBoostM1 prediction for (ECOClib Sergio Escalera)
%##########################################################################

function classes=AdaBoostM1test(data,classifier,params)

classes = predict(classifier,data);
