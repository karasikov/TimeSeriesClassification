function optimal_params = MulticlassSVMTuning(X, y, Parameters, SVM_tuning_parameters)

TRAIN_RATE = 0.7;

if iscell(X)
    [X, y, ~] = CellToMatrixDesign(X, y);
end

classes = unique(y);

train_idx = [];
test_idx = [];

for i = 1 : length(classes)
    c = classes(i);
    idxes = find(y == c);
    test_idx_pos = 1 : length(idxes);
    train_idx_pos = randsample(test_idx_pos, ...
                               floor(TRAIN_RATE * length(test_idx_pos)));
    test_idx_pos(train_idx_pos) = [];
    train_idx = [train_idx; idxes(train_idx_pos)];
    test_idx = [test_idx; idxes(test_idx_pos)];
end

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