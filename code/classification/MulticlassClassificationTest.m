% Test multi-class classification wrapper

function [labels] = MulticlassClassificationTest(test_design_matrix, test_parameters)

[accuracy,labels,Values,confusion]=ECOCTest(test_design_matrix,...
                                            test_parameters.classifiers,...
                                            test_parameters.parameters);

end
