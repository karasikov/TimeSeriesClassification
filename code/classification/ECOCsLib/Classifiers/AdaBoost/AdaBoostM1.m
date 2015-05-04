%##########################################################################
% AdaBoostM1 classifier for (ECOClib Sergio Escalera)
%##########################################################################

function [classifier]=AdaBoostM1(class1, class2, params)

label_vector = [ones(size(class1,1), 1); -1*ones(size(class2,1), 1)];

classifier = fitensemble([class1;class2],label_vector,'AdaBoostM1',params.iterations,'Tree');
