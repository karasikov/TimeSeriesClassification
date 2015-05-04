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

function [thresh,p]=SingleWeakLearnerROC(c1,w1,c2,w2)

V=[c1;c2];
l=size(V,1);
n=size(V,2);
W=[w1 w2]';
valsumW=sum(W);
Y=[ones(length(w1),1); zeros(length(w2),1)];

[V_sort,ind] = sort(V);

YW_p = Y .* W;
YW_n = (1-Y) .* W;
P_sum = sum(YW_p);
N_sum = sum(YW_n);

Cum = YW_p - YW_n;
Cum = cumsum(Cum(ind) / valsumW);

[min1,thresh_ind1] = min(-1*Cum);
min1 = min1 + P_sum*ones(1,n);
[min2,thresh_ind2] = min(Cum);
min2 = min2 + N_sum*ones(1,n);

ind1 = find(min1>min2);
ind2 = find(min1<=min2);
thresh_ind(ind1)=thresh_ind2(ind1);
thresh_ind(ind2)=thresh_ind1(ind2);
thresh_ind(thresh_ind==l)=thresh_ind(thresh_ind==l)-1;

desplazamiento=0:n-1;
thr_ind=thresh_ind+desplazamiento*l;
thresh = (V_sort(thr_ind)+V_sort(thr_ind+1))/2;
p = 2 *(Cum(thr_ind)>0) - 1;
