function optimal_params = MulticlassSVMTuning(X, y, Parameters, SVM_tuning_parameters)

TRAIN_RATE = 0.7;

test_idx = 1 : size(X, 1);
train_idx = randsample(test_idx, round(TRAIN_RATE * size(X, 1)));
test_idx(train_idx) = [];

initialized_settings = Parameters.base_params.settings;

function accuracy = Accuracy(SVM_params)
    Parameters.base_params.settings = [initialized_settings, ' ', ...
        sprintf(strjoin(strcat('-', SVM_tuning_parameters(:,1), ' %f')'), SVM_params)];

    labels = MulticlassClassificationTest(...
                X(test_idx,:), ...
                MulticlassClassificationTrain(X(train_idx,:), y(train_idx), Parameters));

    accuracy = sum(labels == y(test_idx)) / length(test_idx);
    fprintf([Parameters.base_params.settings, ': %f\n'], accuracy);
end

[optimal_params] = ClassifiersTuning(@Accuracy, SVM_tuning_parameters(:,2));

end