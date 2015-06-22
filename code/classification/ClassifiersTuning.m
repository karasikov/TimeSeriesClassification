% Tuning of the classifier's parameters
%
% parameters_ranges --- set of parameters ranges
%
% optimal_params --- [1 x num_params] vector
%
function [optimal_params] = ClassifiersTuning(quality_handle, parameters_ranges)

tested_params = cartprod(parameters_ranges);

quality = zeros(size(tested_params, 1), 1);
fprintf('Current params:\n');
for i = 1 : size(tested_params)
    quality(i) = quality_handle(tested_params(i,:));
end
[~,idx] = max(quality);
fprintf('Best params:\n');
optimal_params = tested_params(idx,:);
disp(optimal_params);
fprintf('Best quality: %f\n', quality(idx));

%% grid of first two parameters
if (length(parameters_ranges) < 2)
    return
end

first  = reshape(tested_params(:,1), [length(parameters_ranges{1}),...
                                      length(parameters_ranges{2})]);
second = reshape(tested_params(:,2), size(first));

%# contour plot of parameter selection
figure
contour(first, second, reshape(quality, size(first))), colorbar
hold on
plot(first(idx), second(idx), 'rx')
text(first(idx), second(idx), sprintf('Quality = %f', quality(idx)), ...
     'HorizontalAlign', 'left', 'VerticalAlign', 'top')
hold off
xlabel('first\_param'), ylabel('second\_param'), title('Parameters tuning')
