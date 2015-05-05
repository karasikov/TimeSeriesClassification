% Train multi-class classification wrapper

function [test_parameters] = MulticlassClassificationTrain(design_matrix, labels, parameters)

[classifiers, parameters]=ECOCTrain(design_matrix, labels, parameters);

test_parameters.classifiers = classifiers;
test_parameters.parameters = parameters;

end
