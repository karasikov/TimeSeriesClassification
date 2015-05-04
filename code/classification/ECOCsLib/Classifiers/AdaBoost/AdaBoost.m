%##########################################################################
% AdaBoost training
%##########################################################################

function [classifier] = AdaBoost(class1, class2, params)

data = sparse([class1;class2]);
labels = [ones(size(class1,1), 1); -1*ones(size(class2,1), 1)];
weights = ones(size(class1,1)+size(class2,1), 1);
T = params.iterations;

for t=1:T
    weights = weights/sum(weights);
    classifier.WeakClassifiers{t} = feval(params.WeakClassifier, labels, data, weights);
    predicted_labels = feval(params.WeakPredictor, classifier.WeakClassifiers{t}, data);
    error = weights' * ((labels .* predicted_labels) < 0);

    if error>=0.5
        return;
    end
    classifier.alpha(t)=log((1-error)/(error+eps))/2;
    weights = weights .* exp(-classifier.alpha(t) * (labels .* predicted_labels));
end
