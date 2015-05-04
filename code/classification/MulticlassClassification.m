% Train multi-class classifier

function [accuracy,Labels,Values,confusion] = TrainClassifier(design_matrix, labels, test_design_matrix, test_labels)

approaches  = {'OneVsAll', 'OneVsOne', 'ECOCRandom', 'ECOCBCH'}; %One-Vs-All, One-Vs-One, ECOC-Random, ECOC-BCH
classifiers = {'SVM', 'SVM', 'ADA'};
tester      = {'SVMtest', 'SVMMtest', 'ADAMtest'};
decoding    = {'HD', 'LLB', 'ELB', 'ED', 'LAP', 'BDEN', 'AED', 'LLW', 'ELW'};

clear Parameters;
Parameters.iterations=1000; %ECOC-Random parameter
Parameters.columns=18; %ECOC-Random parameter: code length
Parameters.BCHcodelength=15; %ECOC-BCH parameter: code length
Parameters.decoding='HD'; %Hamming distance
Parameters.base_params.iterations=50; %for AdaBoost binary classifier
Parameters.base_params.settings='-c 8 -g 0.0078'; %for SVM binary classifier

k=1;
j=2;

Parameters.base=classifiers{k};
Parameters.base_test=tester{k};
Parameters.decoding=decoding{k};
Parameters.coding=approaches{j};

[Classifiers, Parameters]=ECOCTrain(design_matrix,...
                                    labels,...
                                    Parameters);

[accuracy,Labels,Values,confusion]=ECOCTest(test_design_matrix,...
                                            Classifiers,...
                                            Parameters,...
                                            test_labels);
end
