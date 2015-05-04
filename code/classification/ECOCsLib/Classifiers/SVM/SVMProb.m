%##########################################################################
% SVM-Probability classifier for (ECOClib Sergio Escalera)
%##########################################################################

function [classifier]=SVMProb(class1, class2, params)

label_vector = [ones(size(class1,1), 1); -1*ones(size(class2,1), 1)];

classifier = svmtrain(label_vector, [class1; class2], [params.settings ' -b 1 -q']);
