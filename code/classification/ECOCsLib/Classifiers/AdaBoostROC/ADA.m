%##########################################################################

% <ECOCs Library. Coding and decoding designs for multi-class problems.>
% Copyright (C) 2009 Sergio Escalera sergio@maia.ub.es

%##########################################################################

% This file is part of the ECOC Library.

% ECOC Library is free software; you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free Software 
% Foundation; either version 2 of the License, or (at your option) any later version. 

% This program is distributed in the hope that it will be useful, but WITHOUT ANY 
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR 
% A PARTICULAR PURPOSE. See the GNU General Public License for more details. 

% You should have received a copy of the GNU General Public License along with
% this program. If not, see <http://www.gnu.org/licences/>.

%##########################################################################

function [classifier]=ADA(clase1, clase2, params, test)

T=params.iterations;

Nc1=size(clase1,1);
Nc2=size(clase2,1);
w1=ones(1,Nc1);
w2=ones(1,Nc2);
classifier.feature = zeros(T,1);
classifier.thr = zeros(T,1);
classifier.p = zeros(T,1);
classifier.alpha=[];

for t=1:T
    tw=sum(w1)+sum(w2);
    w1=w1/tw;
    w2=w2/tw;
    [weak.thr,weak.p] = SingleWeakLearnerROC(clase1,w1,clase2,w2);
    error=weak.p.*(w1*(clase1 >= repmat(weak.thr,Nc1,1))) + ...
          weak.p.*(w2*(clase2 <  repmat(weak.thr,Nc2,1))) -(weak.p-1)/2;
    [err,featureO]=min(error);
    if err>=0.5
        return;
    end
    classifier.feature(t)=featureO;
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