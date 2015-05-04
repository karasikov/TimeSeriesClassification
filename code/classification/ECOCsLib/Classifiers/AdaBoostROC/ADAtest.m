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

function classes=ADAtest(data,classifier,params)

T=length(classifier.alpha);
if T==0
    classes=zeros(size(data,1),1);
    return;
end

accum_result=zeros(size(data,1),1);
thresh=0;
for i=1:T
    accum_result=accum_result+classifier.alpha(i)*(classifier.p(i)*data(:,classifier.feature(i)) < classifier.p(i)*classifier.thr(i));
    thresh=thresh+classifier.alpha(i);
end;
classes=accum_result>=thresh/2;
classes = 2*classes - 1;
