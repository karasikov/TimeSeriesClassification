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

function [Classifiers,Parameters]=ECOCTrain(data,labels,Parameters)

%##########################################################################

if (nargin<2)
    error('Exit: Incorrect number of parameters to function ECOCTrain.');
end

% Parameters
if isfield(Parameters,'coding')==0
    disp('No coding defined, using One-Versus-One');
    Parameters.coding='OneVsOne';
end
if isfield(Parameters,'decoding')==0
    disp('No decoding defined, using Hamming decoding');
    Parameters.decoding='HD';
end
if isfield(Parameters,'show_info')==0
    Parameters.show_info=0;
end
if isfield(Parameters,'store_training_data')==0
    Parameters.store_training_data=0;
end
if isfield(Parameters,'ECOC')==0
    Parameters.ECOC=[];
end

%     Coding Sparse
if isfield(Parameters,'columns')==0
    Parameters.columns=15;
end
if isfield(Parameters,'iterations')==0
    Parameters.iterations=3000;
end
if isfield(Parameters,'zero_prob')==0
    Parameters.zero_prob=0.5;
end

%         BCH
if isfield(Parameters,'BCHcodelength')==0
    Parameters.BCHcodelength=15;
end

%       ECOC-ONE
if isfield(Parameters,'validation')==0
    Parameters.validation=0.15;
end
if isfield(Parameters,'w_validation')==0
    Parameters.w_validation=0.5;
end
if isfield(Parameters,'epsilon')==0
    Parameters.epsilon=0.05;
end
if isfield(Parameters,'iterations_one')==0
    Parameters.iterations_one=5;
end
if isfield(Parameters,'steps_one')==0
    Parameters.steps_one=5;
end
if isfield(Parameters,'ECOC_initial')==0
    Parameters.ECOC_initial='OneVsAll';
end
if isfield(Parameters,'one_mode')==0
    Parameters.one_mode=2;
end
if isfield(Parameters,'iterations_sffs')==0
    Parameters.iterations_sffs=5;
end
if isfield(Parameters,'steps_sffs')==0
    Parameters.steps_sffs=5;
end
if isfield(Parameters,'criterion_sffs')==0
    Parameters.criterion_sffs=1;
end

%       Forest-ONE
if isfield(Parameters,'number_trees')==0
    Parameters.number_trees=3;
end

%     Base classifier
if isfield(Parameters,'base')==0
    error('No base classifier defined');
end
if isfield(Parameters,'base_params')==0
    Parameters.base_params='';
end
if isfield(Parameters,'base_test')==0
    Parameters.base_test='';
end
if isfield(Parameters,'base_test_params')==0
    Parameters.base_test_params='';
end

%     Custom coding
if isfield(Parameters,'custom_coding')==0
    Parameters.custom_coding='';
end
if isfield(Parameters,'custom_coding_params')==0
    Parameters.custom_coding_params='';
end
%   Custom decoding
if isfield(Parameters,'custom_decoding')==0
    Parameters.custom_decoding='';
end
if isfield(Parameters,'custom_decoding_params')==0
    Parameters.custom_decoding_params='';
end


if length(labels)~=size(data,1)
    error('Wrong labels size');
end

try,
    data(:,end+1)=labels;
catch,
    data(:,end+1)=labels';
end

show_info=Parameters.show_info;

if (size(data,2)<2)
    error('Exit: Incorrect size of data.');
end

classes=unique(data(:,end)); % look for the number of classes
number_classes=length(classes);

if (number_classes<2)
    error('Exit: Incorrect number of classes.');
end

switch Parameters.coding, % Define the coding matrix
    case 'OneVsOne', % one-versus-one design
        ECOC=OneVsOne(number_classes);
    case 'OneVsAll', % one-versus-all design
        ECOC=OneVsAll(number_classes);
    case 'ECOCRandom', % Sparse design
        if show_info
            disp(['Computing sparse coding matrix'])
        end
        ECOC=Coding_sparse(classes,Parameters.columns,Parameters.iterations,Parameters.zero_prob);
    case 'ECOCBCH', % BCH design
        ECOC=BCH(number_classes,2^ceil(log2(Parameters.BCHcodelength+1))-1);
    case 'AllPairs', % All-Pairs design
        l=size(data,1);
        avgs=zeros(number_classes,size(data,2)-1);
        new_data=zeros(l*number_classes,size(data,2)-1);
        new_labels=-1*ones(size(new_data,1),1);
        for i=1:number_classes
            idx=find(data(:,end)==classes(i));
            avgs(i,:)=sum(data(idx,1:end-1))/length(idx);
            new_data(l*(i-1)+1:l*i,:)=data(:,1:end-1)-repmat(avgs(i,:),size(data,1),1);
            new_labels(idx+l*(i-1))=1;
        end
        Parameters.ECOC=[1,1;-1,-1];
        Classifiers=Learning([new_data,new_labels],Parameters);
        Parameters.ECOC=[OneVsAll(number_classes),avgs,classes];
        return;
    case 'ECOCONE', % ECOC-ONE design
        try
            if ((Parameters.validation<=0) || (Parameters.validation>=1))
                error('Exit: Size of validation subset must be between (0,1) for the ecoc-one coding design.');
            end
        catch
            error('Exit: Parameters.validation bad defined.');
        end
        try
            if ((Parameters.w_validation<=0) || (Parameters.w_validation>=1))
                error('Exit: Weight of validation subset must be between (0,1) for the ecoc-one coding design.');
            end
        catch
            error('Exit: Parameters.w_validation bad defined.');
        end
        try
            if ((Parameters.epsilon<0) || (Parameters.epsilon>1))
                error('Exit: Value of epsilon must be between 0 and 1 for the ecoc-one coding design.');
            end
        catch
            error('Exit: Parameters.epsilon bad defined.');
        end
        switch Parameters.ECOC_initial, % Define the initial coding matrix for ECOC-ONE
            case 'OneVsOne', % one-versus-one design
                ECOC=OneVsOne(number_classes);
            case 'OneVsAll', % one-versus-all design
                ECOC=OneVsAll(number_classes);
            case 'ECOCRandom', % Sparse design
                if show_info
                    disp(['Computing sparse coding matrix for ecoc-one initialization'])
                end
                ECOC=Coding_sparse(classes,Parameters.columns,Parameters.iterations,Parameters.zero_prob);
            case 'CUSTOM',
                if show_info
                    disp(['Computing custom coding matrix'])
                end
                try,
                    ECOC=feval(Parameters.custom_coding,classes,Parameters.custom_coding_params);
                catch,
                    error('Exit: Coding error when using custom coding strategy.');
                end
        end
        if (size(ECOC,2)<1)
            error('Exit: At least an initial one-column matrix is required for the ecoc-one coding design.');
        end
        Parameters.ECOC=ECOC;
        ECOC=ecoc_one(classes,data,Parameters,Parameters.decoding,show_info);
    case 'DECOC', % DECOC design
        if show_info
            disp(['Computing DECOC coding matrix'])
        end
        ECOC=DECOC(classes,data,Parameters.criterion_sffs,Parameters.iterations_sffs,show_info);
    case 'Forest', % Forest-ECOC design
        if show_info
            disp(['Computing Forest-ECOC coding matrix'])
        end
        ECOC=Forest(classes,data,Parameters.criterion_sffs,Parameters.iterations_sffs,Parameters.number_trees,show_info);
    case 'CUSTOM', % Using custom coding function
        if show_info
            disp(['Computing custom coding matrix'])
        end
        try,
            ECOC=feval(Parameters.custom_coding,classes,Parameters.custom_coding_params);
        catch,
            error('Exit: Coding error when using custom coding strategy.');
        end
    otherwise,
        error('Exit: Coding design bad defined.');
end
if show_info
    disp(['Learning coding matrix :']);
    disp(ECOC);
end

Parameters.ECOC=[ECOC, classes];
Classifiers=Learning(data,Parameters); % Call for training the data with the defined coding design and base classifier
