function [dataset] = load_USCHAD_dataset()

if (length(mfilename()))
    cur_dir = fileparts(which(mfilename()));
else
    cur_dir = pwd;
end
path_to_data = strcat(cur_dir, '/../../../data/USC-HAD/');

try
    load([path_to_data, 'USC_HAD_dataset.mat'], 'dataset');
catch
    if ~length(ls([path_to_data, '\Subject*']))
        addpath([path_to_data, '/../']);
        GetDatasetUSCHAD();
    end

    subjects = ls([path_to_data, '\Subject*']);

    dataset = struct([]);

    for i = 1:length(subjects)
        timeseries = ls(strcat(path_to_data, subjects(i,:), '\*mat'));
        for j = 1:length(timeseries)
            try
                load( strcat( path_to_data, subjects(i,:), '\', timeseries(j,:) ), 'sensor_readings', 'activity_number' );
                new_observation.label = str2num(activity_number);
                clear activity_number
            catch e
                load( strcat( path_to_data, subjects(i,:), '\', timeseries(j,:) ), 'sensor_readings', 'activity_numbr' );
                new_observation.label = str2num(activity_numbr);
                clear activity_numbr
            end
            new_observation.ts = sensor_readings';
            dataset = [dataset, new_observation];
        end
    end

    save([path_to_data, 'USC_HAD_dataset.mat'], 'dataset');
end

end