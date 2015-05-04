% USC_HAD_dataset

function dataset = GetDatasetUSCHAD

if ~exist('..\data\USC-HAD\USC_HAD_dataset.mat', 'file')
    if ~exist('..\data\USC-HAD.zip', 'file')
        urlwrite('http://sipi.usc.edu/HAD/USC-HAD.zip', '..\data\USC-HAD.zip')
    end

    if ~length(ls('..\data\USC-HAD\Subject*'))
        unzip('..\data\USC-HAD.zip')
    end

    subjects = ls('..\data\USC-HAD\Subject*');

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

    save('..\data\USC-HAD\USC_HAD_dataset.mat', 'dataset');
end

load('..\data\USC-HAD\USC_HAD_dataset.mat', 'dataset');

end