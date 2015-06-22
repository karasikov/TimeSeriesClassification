function [dataset] = aggregate(dataset, window_size)

for i = 1 : length(dataset)
    ts = dataset(i).ts;
    dataset(i).ts = [];
    for j = 1 : floor(size(ts, 2) / window_size)
        dataset(i).ts(:, j) = mean(...
            ts(:, window_size * (j - 1) + 1 : window_size * j),...
            2);
    end
end

end