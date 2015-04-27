% USC_HAD_dataset

if ~exist('USC-HAD\USC_HAD_dataset.mat', 'file')
    if ~exist('USC-HAD.zip', 'file')
        urlwrite('http://sipi.usc.edu/HAD/USC-HAD.zip', 'USC-HAD.zip')
    end

    if ~length(ls('USC-HAD\Subject*'))
        unzip('USC-HAD.zip')
    end

    subjects = ls('USC-HAD\Subject*');

    dataset = {};

    for i = 1:length(subjects)
        timeseries = ls(strcat('USC-HAD\', subjects(i,:), '\*mat'));
        for j = 1:length(timeseries)
            try
                load(strcat('USC-HAD\', subjects(i,:), '\', timeseries(j,:)), 'sensor_readings', 'activity_number');
                dataset{end + 1} = struct('ts', sensor_readings, 'class', str2num(activity_number));
                clear sensor_readings activity_number
            catch e
                load(strcat('USC-HAD\', subjects(i,:), '\', timeseries(j,:)), 'sensor_readings', 'activity_numbr');
                dataset{end + 1} = struct('ts', sensor_readings, 'class', str2num(activity_numbr));
                clear sensor_readings activity_numbr
            end
        end
    end

    save('USC-HAD\USC_HAD_dataset.mat', 'dataset')
end
