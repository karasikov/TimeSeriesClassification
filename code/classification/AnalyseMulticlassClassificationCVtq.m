function [ classes_distr, sensitivity ] = AnalyseMulticlassClassification( ...
                                            X, y, learn_fun, par_learn, ...
                                            test_fun, nfolds, nsplits, ...
                                            accuracy_figure_name)
% The function provides multi-class classification analysis through
% (nsplits, nfolds)-fold cross-validation
% It splits sample on the nfolds folds 'nsplits' times, estimates
% classifier parameters on a learn part,
% and estimates accuracy on a test part.
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
% nfolds - learn/test splitting size ratio is (nfolds - 1) : (1), e.g. 3
% nsplits - number of splittings, e.g. 100
% accuracy_figure_name - name of saving figure
%
% Output:
% classes_distr - K-by-K confusion matrix http://en.wikipedia.org/wiki/Confusion_matrix
% sensitivity   - K-by-1 vector of TPR for each class

classes = unique(y);
classes_distr = zeros(length(classes));

fprintf('Multi-class classification analysis, %d splits. Current split:\n', nsplits);
% Splitting cycle
for split_idx = 1 : nsplits
    cv_indeces = crossvalind('Kfold', length(y), nfolds);
    for training_fold_idx = 1 : nfolds
        fprintf('%d..', split_idx);
        %% Sample train-test splitting
        train_idx = cv_indeces ~= training_fold_idx;
        test_idx = cv_indeces == training_fold_idx;

        [X_train, y_train, ~] = CellToMatrixDesign(X(train_idx), y(train_idx));
        [X_test, ~, observation_indexes] = CellToMatrixDesign(X(test_idx), y(test_idx));
        y_test = y(test_idx);

        %% Train classifier
        par = feval(learn_fun, X_train, y_train, par_learn);

        %% Classify samples
        y_fragments_est_test = feval(test_fun, X_test, par);
        y_est_test = zeros(size(y_test));
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
end
fprintf('\nDone\n');

classes_distr = classes_distr / nsplits;
num_obj = sum(classes_distr, 2);

sensitivity = diag(classes_distr) ./ num_obj;

%% bar chart with quality
h = figure; hold on;
bar(classes, num_obj)
bar(classes, num_obj .* sensitivity, 'g')
axis('tight');
ylim([0, max(num_obj) * 1.07]);
xlabel('Class labels', ...
       'FontSize', 20, 'FontName', 'Times', 'Interpreter', 'latex');
ylabel('Objects number', ...
       'FontSize', 20, 'FontName', 'Times', 'Interpreter', 'latex');
set(gca, 'FontSize', 20, 'FontName', 'Times');
text(classes, num_obj + ceil(0.01 * max(num_obj)), ...
     num2str(sensitivity * 100, '%0.1f\\%%'), ...
     'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'bottom', ...
     'FontSize', ceil(19 * 6 / length(num_obj)), ...
     'FontName', 'Times', 'Interpreter', 'latex');
title(['Mean accuracy: ', ...
       num2str(sum(diag(classes_distr)) / sum(num_obj), '%0.4f')], ...
      'FontSize', 22, 'FontName', 'Times', 'Interpreter', 'latex');

if nargin >= 8
    saveas(h,[accuracy_figure_name '.eps'], 'psc2');
    saveas(h,[accuracy_figure_name '.png'], 'png');
end

%% confusion matrix to tex
num_classes = length(classes);
confusion = classes_distr ./ repmat(num_obj, 1, num_classes);
latex_table = sprintf(['\\begin{tabular}{|', repmat('c|', 1, 2 + num_classes), '}\n',...
                       '  \\cline{3-%d}\n'...
                       '  \\multicolumn{2}{c|}{} & \\multicolumn{%d}{c|}{Predicted class} \\\\ \\cline{3-%d}\n'],...
                       2 + num_classes, num_classes, 2 + num_classes);
latex_table = [latex_table,...
    '  \multicolumn{2}{c|}{} ', num2str(classes', ' & $%d$'), sprintf('\\\\ \\cline{1-%d}\n', 2 + num_classes)];
latex_table = [latex_table,...
    sprintf('  \\multirow{%d}{*}{\\begin{sideways}Actual class\\end{sideways}}\n', num_classes)];

for i = 1 : num_classes - 1
    latex_table = [latex_table, sprintf(['  & $%d$ ', num2str(confusion(i, :), ' & $%.2f$'), '\\\\ \\cline{2-%d}\n'], classes(i), 2 + num_classes)];
end
latex_table = [latex_table, sprintf(['  & $%d$ ', num2str(confusion(end, :), ' & $%.2f$'), '\\\\ \\cline{1-%d}\n\\end{tabular}\n'], classes(end), 2 + num_classes)];

if nargin >= 8
    fid = fopen([accuracy_figure_name '.tex'], 'w');
    fprintf(fid, '%s', latex_table);
    fclose(fid);
end

end