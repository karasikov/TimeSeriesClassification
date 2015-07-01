%% add path
if (length(mfilename()))
    cur_dir = fileparts(which(mfilename()));
else
    cur_dir = pwd;
end
addpath(genpath(strcat(cur_dir, '/../classification')));
addpath(genpath(strcat(cur_dir, '/../data_loading')));
addpath(genpath(strcat(cur_dir, '/../feature_generation')));
addpath(genpath(strcat(cur_dir, '/../segmentation')));

%% feature generation
% statistical features for one-component ts: [1 x ts_len] double
single_features = {...
    @(ts)( mean(ts) ),...
    @(ts)( std(ts) ),...
    @(ts)( mean(abs(ts - mean(ts))) ),...
    @(ts)( CalcAR(ts, 10) ),...
};

% tses: [num_components x ts_len] double
multi_features = {...
    @(tses)( CalcAR(sqrt(sum(tses(1:3,:).^2, 1)), 10) ),...
    @(tses)( CalcAR(sqrt(sum(tses(4:6,:).^2, 1)), 10) ),...
};

% smoothing: round each 10 measurements
dataset = aggregate(load_USCHAD_dataset(), 10);

dataset = dataset([dataset.label] <= 10);

segm_size = round(linspace(30, length(dataset(1).ts(1,:)) / 2, 10));
mean_accuracy = zeros(size(segm_size));

for segm_size_idx = 1 : length(segm_size)
    [X,y] = GenerateFeatures(dataset, single_features, multi_features, ...
                             @(tses)( random_segments(tses, segm_size(segm_size_idx), 20) ), ...
                             @(fragments_features)( fragments_features ));
    X = ScaleCell(X);

    %% Multi-class classification settings
    Parameters.coding = 'OneVsOne';
    Parameters.decoding = 'HD';
    Parameters.base = 'SVMLinear';
    Parameters.base_test = 'SVMLinearTest';

    % MulticlassSVMTuning(X, y, Parameters, {'c', 35:5:60; 'g', 0.0005:0.0005:0.004});
    Parameters.base_params.settings='-c 0.25'; %for SVM binary classifier

    %% Analyze classification
    % Determine learn/test splitting parameters
    NSPLITS = 5;
    LEARN_RATE = 0.7;
    % Launch analyzer
    [confusion,sens] = AnalyseMulticlassClassification(X, y, ...
                            @MulticlassClassificationTrain, Parameters, ...
                            @MulticlassClassificationTest, ...
                            LEARN_RATE, NSPLITS);
    disp(sens');

    mean_accuracy(segm_size_idx) = sum(diag(confusion)) / sum(confusion(:));
end

accuracy_figure_name = ['VotingSegments_Dataset_USCHAD_full' ...
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

h = figure; hold on; grid on;
plot(segm_size, [mean_accuracy; mean_accuracy_], 'Linewidth', 1.5);
title('Voting Segments vs Normal Distribution', ...
      'FontSize', 20, 'FontName', 'Times', 'Interpreter', 'latex');
xlabel('Size of segments', 'FontSize', 20, 'FontName', 'Times', 'Interpreter', 'latex');
ylabel('Mean accuracy', 'FontSize', 20, 'FontName', 'Times', 'Interpreter', 'latex');
legend({'Voting Segments', 'Normal distribution'}', ...
       'Location','NorthWest',...
       'FontSize',22,'FontName','Times','Interpreter','latex');
legend('boxoff');
set(gca, 'xdir', 'reverse', 'FontSize', 18, 'FontName', 'Times');
saveas(h,['NormalDistributionVS' accuracy_figure_name '.eps'],'psc2');
saveas(h,['NormalDistributionVS' accuracy_figure_name '.png'],'png');
