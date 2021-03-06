%##########################################################################
% AdaBoost prediction for implementation with reducing dimensions
%##########################################################################

function classes=ADARDtest(data,classifier,params)

T=length(classifier.alpha);
if T==0
    classes=zeros(size(data,1),1);
    return;
end
accum_result=zeros(size(data,1),1);
thresh=0;
projected_data=data*classifier.proj;
for i=1:T
    accum_result=accum_result+classifier.alpha(i)*(classifier.p(i)*projected_data(:,i) < classifier.p(i)*classifier.thr(i));
    thresh=thresh+classifier.alpha(i);
end;
classes=accum_result>=thresh/2;
classes = 2*classes - 1;