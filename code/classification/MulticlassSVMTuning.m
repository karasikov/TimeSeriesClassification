function optimal_params = MulticlassSVMTuning(X, y, Parameters, SVM_tuning_parameters)

TRAIN_RATE = 0.7;

test_idx = 1 : size(X, 1);
train_idx = randsample(test_idx, round(TRAIN_RATE * size(X, 1)));
test_idx(train_idx) = [];

initialized_settings = Parameters.base_params.settings;

function test_accuracy = Accuracy(SVM_params)
    Parameters.base_params.settings = [initialized_settings, ' ', ...
        sprintf(strjoin(strcat('-', SVM_tuning_parameters(:,1), ' %f')'), SVM_params)];

    classifier = MulticlassClassificationTrain(X(train_idx,:), y(train_idx), Parameters);
    train_labels = MulticlassClassificationTest(X(train_idx,:), classifier);
    test_labels = MulticlassClassificationTest(X(test_idx,:), classifier);

    train_accuracy = sum(train_labels == y(train_idx)) / length(train_idx);
    test_accuracy = sum(test_labels == y(test_idx)) / length(test_idx);
    fprintf([Parameters.base_params.settings, ': %f/%f\n'], train_accuracy, test_accuracy);
end

[optimal_params] = ClassifiersTuning(@Accuracy, SVM_tuning_parameters(:,2));

end