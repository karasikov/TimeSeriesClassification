function [ classes_distr, sensitivity ] = AnalyseMulticlassClassification( ...
                                            X, y, learn_fun, par_learn, ...
                                            test_fun, learn_rate, nsplits )
% The function provides multi-class classification analysis.
% It splits sample on the learn and test parts 'nsplits' times, estimates
% classifier parameters on a learn part, and calculates error values on a
% test part.
%
% Inputs:
% X - objects-features matrix, (m x 1) cell of (p x n) matrices, p --- number of observations per object
% y - target vector (m x 1),
% learn_fun - handle on a learn classifier function, e.g. @learnfun
%   (note that learn_fun must receive 3 parameters: X, y and structure of all necessary learning parameters)
% par_learn - structure of all necessary learning parameters, 3-rd argument
%   of learn_fun
% test_fun - handle on a learn classifier function, e.g. @testfun
%   (note that learn_fun must receive 2 parameters: X_test and output of the learn_fun)
% learn_rate - learn/test splitting size ratio, e.g. 0.7
% nsplits - number of splittings, e.g. 100
%
% Output:
% classes_distr - K-by-K confusion matrix http://en.wikipedia.org/wiki/Confusion_matrix
% sensitivity   - K-by-1 vector of TPR for each class


classes = unique(y);
train_size = round(learn_rate * length(y));
classes_distr = zeros(length(classes));

fprintf('Multi-class classification analysis, %d splits. Current split:\n', nsplits);
% Splitting cycle
for split_idx = 1 : nsplits
    fprintf('%d..', split_idx);
    %% Sample train-test splitting
    perm_idx = randperm(length(y));

    train_idx = perm_idx(1 : train_size);
    test_idx = perm_idx(train_size + 1 : length(y));

    [X_train, y_train, ~] = Reshape(X(train_idx), y(train_idx));
    [X_test, ~, observation_indexes] = Reshape(X(test_idx), y(test_idx));
    y_test = y(test_idx);

    %% Train classifier
    par = feval(learn_fun, X_train, y_train, par_learn);

    %% Classify samples
    y_fragments_est_test = feval(test_fun, X_test, par);
    y_est_test = zeros(size(test_idx));
    for i = 1 : length(y_est_test) % for each testing observation
        y_est_test(i) = mode(y_fragments_est_test(observation_indexes == i));
    end

    %% Confusion matrix
    for cl1 = 1 : length(classes)
        for cl2 = 1 : length(classes)
            y_cl1 = find(y_test == classes(cl1));
            y_cl2 = find(y_est_test == classes(cl2));
            classes_distr(cl1, cl2) = classes_distr(cl1, cl2) + ...
                                      length(intersect(y_cl1, y_cl2));
        end
    end
end
fprintf('\nDone\n');

sensitivity = diag(classes_distr) ./ sum(classes_distr, 2);
classes_distr = classes_distr / nsplits;

num_obj = sum(classes_distr, 2);
mean_test = diag(classes_distr) ./ num_obj;

figure
hold on
bar(num_obj)
bar(num_obj .* mean_test, 'g')
axis('tight');
xlabel('Class labels', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Objects number', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 20, 'FontName', 'Times')
title(['Mean test quality=',num2str(sum(diag(classes_distr)) / sum(num_obj))])

end


function [X_out, y_out, observation_indexes] = Reshape(X, y)

X_out = cell2mat(X);

y_out = [];
observation_indexes = [];
for i = 1 : length(X)
    num_observations = size(X{i}, 1);
    y_out = [y_out; repmat(y(i), num_observations, 1)];
    observation_indexes = [observation_indexes; repmat(i, num_observations, 1)];
end

end