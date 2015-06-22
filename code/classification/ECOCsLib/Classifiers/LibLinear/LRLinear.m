%##########################################################################
% LibLinear LR classifier for (ECOClib Sergio Escalera)
%##########################################################################

function [classifier] = SVMLinear(class1, class2, params)

label_vector = [ones(size(class1,1), 1); -1*ones(size(class2,1), 1)];

classifier = liblineartrain(label_vector, sparse([class1; class2]), ...
                            ['-s 0 ' params.settings ' -q']);

end