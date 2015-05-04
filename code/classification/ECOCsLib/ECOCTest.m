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

function [result,Labels,Values,confusion]=ECOCTest(TestData,Classifiers,Parameters,labels)

%##########################################################################

if (nargin<3)
    error('Exit: Incorrect number of parameters to function ECOCTest.');
elseif nargin~=4
    labels=zeros([1 size(TestData,1)]);
end

if isfield(Parameters,'show_info')==0
    Parameters.show_info=0;
end
if isfield(Parameters,'ECOC')==0
    Parameters.ECOC=[];
end
if isfield(Parameters,'base_test')==0
    Parameters.base_test='';
end
if isfield(Parameters,'base_test_params')==0
    Parameters.base_test_params='';
end
%   Custom decoding
if isfield(Parameters,'custom_decoding')==0
    Parameters.custom_decoding='';
end
if isfield(Parameters,'custom_decoding_params')==0
    Parameters.custom_decoding_params='';
end

if length(labels)~=size(TestData,1)
    error('Wrong labels size');
end

try,
    TestData(:,end+1)=labels;
catch,
    TestData(:,end+1)=labels';
end


if size(TestData,1)~=0
    if Parameters.show_info
        disp(['Testing ECOC design'])
    end

    if size(Parameters.ECOC,1)==0
        error('Exit: ECOC matrix not defined.');
    end

    % call to the corresponding decoding strategy with the current ecoc configuration
    [result,confusion,Labels,Values]=Decoding(TestData,Parameters.ECOC(:,end),Parameters.ECOC(:,1:end-1),Parameters.base,Classifiers,Parameters.decoding,Parameters.base_test,Parameters.base_test_params,Parameters.custom_decoding,Parameters.custom_decoding_params);
else
    Labels=[];
    Values=[];
end