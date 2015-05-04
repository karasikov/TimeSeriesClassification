%##########################################################################
% AdaBoost with reducing dimensions
% Base algorithm: Decision stump
% Base implementation: <ECOCs Library. Sergio Escalera>
%##########################################################################

function [classifier]=ADARD(class1, class2, params)

dim=size(class1,2);
proj_dim=params.proj_dim;
T=params.iterations;

Nc1=size(class1,1);
Nc2=size(class2,1);
w1=ones(1,Nc1);
w2=ones(1,Nc2);
classifier.proj = zeros(dim,T);
classifier.thr = zeros(T,1);
classifier.p = zeros(T,1);
classifier.alpha=[];

for t=1:T
    tw=sum(w1)+sum(w2);
    w1=w1/tw;
    w2=w2/tw;

    G=randn(dim,proj_dim)/sqrt(proj_dim);
    clase1 = class1 * G;
    clase2 = class2 * G;

    [weak.thr,weak.p] = SingleWeakLearnerROC(clase1,w1,clase2,w2);
    error=weak.p.*(w1*(clase1 >= repmat(weak.thr,Nc1,1))) + ...
          weak.p.*(w2*(clase2 <  repmat(weak.thr,Nc2,1))) -(weak.p-1)/2;
    [err,featureO]=min(error);
    if err>=0.5
        return;
    end
    classifier.proj(:,t)=G(:,featureO);
    classifier.thr(t)=weak.thr(featureO);
    classifier.p(t)=weak.p(featureO);
    betaO=err/(1-err);
    classifier.alpha(t)=-log10(betaO+eps);
    fe= (classifier.p(t)*clase1(:,featureO) < classifier.p(t)*classifier.thr(t));
    w1=w1.*(betaO.^(fe))';
    nfe= (classifier.p(t)*clase2(:,featureO) >= classifier.p(t)*classifier.thr(t)); 
    w2=w2.*(betaO.^(nfe))';
    if err<eps
        return;
    end
end