%% add path
if (length(mfilename()))
    cur_dir = fileparts(which(mfilename()));
else
    cur_dir = pwd;
end
addpath(genpath(strcat(cur_dir, '/../classification')));
addpath(genpath(strcat(cur_dir, '/../data_loading')));
addpath(genpath(strcat(cur_dir, '/../feature_generation')));

%% feature generation
% statistical features for one-component ts: [1 x ts_len] double
single_features = {...
    @(ts)( mean(ts) ),...
    @(ts)( std(ts) ),...
    @(ts)( mean(abs(ts - mean(ts))) ),...
    @(ts)( CalcAR(ts, 10) ),...
...
};

% tses: [num_components x ts_len] double
multi_features = {...
    @(tses)( CalcAR(sqrt(sum(tses(1:3,:).^2, 1)), 10) ),...
    @(tses)( CalcAR(sqrt(sum(tses(4:6,:).^2, 1)), 10) ),...
};

dataset = aggregate(load_USCHAD_dataset(), 10);

dataset = dataset([dataset.label] <= 10);

num_segments = 1:9;
segm_size = round(length(dataset(1).ts(1,:)) ./ num_segments);
mean_accuracy = zeros(size(segm_size));

for segm_size_idx = 1 : length(segm_size)
    [X,y] = GenerateFeatures(dataset, single_features, multi_features, ...
                             @(tses)( partition(tses, segm_size(segm_size_idx)) ), ...
                             @(fragments_features)( fragments_features ));
    X = ScaleCell(X);

    %% Multi-class classification settings
    Parameters.coding = 'OneVsOne';
    Parameters.decoding = 'HD';
    Parameters.base = 'SVMLinear';
    Parameters.base_test = 'SVMLinearTest';

    % MulticlassSVMTuning(X, y, Parameters, {'c', 35:5:60; 'g', 0.0005:0.0005:0.004});

    Parameters.base_params.settings='-c 1'; %for SVM binary classifier

    %% Analyze classification
    % Determine learn/test splitting parameters
    NSPLITS = 50;
    LEARN_RATE = 0.7;
    % Launch analyzer
    [confusion,sens] = AnalyseMulticlassClassification(X, y, ...
                            @MulticlassClassificationTrain, Parameters, ...
                            @MulticlassClassificationTest, ...
                            LEARN_RATE, NSPLITS);
    disp(sens');

    mean_accuracy(segm_size_idx) = sum(diag(confusion)) / sum(confusion(:));
end

accuracy_figure_name = ['VotingSegments_Dataset_USCHAD' ...
                        '_nSplits_' num2str(NSPLITS) ...
                        '_rate_' num2str(LEARN_RATE) ...
                        '_approach_' Parameters.coding ...
                        '_' Parameters.decoding ...
                        '_classifier_' Parameters.base ...
                        '_' Parameters.base_params.settings];
accuracy_figure_name = regexprep(accuracy_figure_name, ' ', '');

h = figure; hold on; grid on;
plot(segm_size, mean_accuracy, 'Linewidth', 1.5);
title('Voting Segments', 'FontSize', 20, 'FontName', 'Times', 'Interpreter', 'latex');
xlabel('Size of segments', 'FontSize', 20, 'FontName', 'Times', 'Interpreter', 'latex');
ylabel('Mean accuracy', 'FontSize', 20, 'FontName', 'Times', 'Interpreter', 'latex');
set(gca, 'xdir', 'reverse', 'FontSize', 18, 'FontName', 'Times');
saveas(h,[accuracy_figure_name '.eps'], 'psc2');
saveas(h,[accuracy_figure_name '.png'], 'png');
