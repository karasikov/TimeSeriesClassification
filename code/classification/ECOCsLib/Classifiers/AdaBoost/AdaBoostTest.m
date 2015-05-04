%##########################################################################
% AdaBoost prediction interface
%##########################################################################

function classes=AdaBoostTest(data,classifier,params)

T = length(classifier.alpha);
sum = zeros(size(data,1),1);

for t=1:T
    sum = sum + classifier.alpha(t)*feval(params.WeakPredictor, classifier.WeakClassifiers{t}, data);
end
classes = sign(sum);
