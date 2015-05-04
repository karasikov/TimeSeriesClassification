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

function Classifiers=Learning(data,Parameters)

number_of_classifiers = size(Parameters.ECOC,2)-1;
classes = Parameters.ECOC(:,end);

Classifiers=[];
for i=1:number_of_classifiers % for each column of the coding matrix
    %reducing to binary problems {-1,+1}
    p_classes=classes(find(Parameters.ECOC(:,i)==1));
    FirstSet=data(ismember(data(:,end),p_classes),1:end-1);
    n_classes=classes(find(Parameters.ECOC(:,i)==-1));
    SecondSet=data(ismember(data(:,end),n_classes),1:end-1);
    %try, % apply the base classifier over the current partition of classes
        Classifiers{i}.classifier=feval(Parameters.base,FirstSet,SecondSet,Parameters.base_params);
    %catch,
    %    error('Exit: Base classifier bad defined for custom base classifier.');
    %end
    if Parameters.store_training_data
        Classifiers{i}.FirstSet=FirstSet;
        Classifiers{i}.SecondSet=SecondSet;
    end
end
