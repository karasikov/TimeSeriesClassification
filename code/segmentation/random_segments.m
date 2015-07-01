% generate |num_segments| random segments of size |segm_size|
%
function [parts] = partition(tses, segm_size, num_segments)

n = size(tses, 2);
if n < segm_size
    segm_size = n;
end

begins = randsample(n - segm_size + 1, num_segments, true);

for i = 1 : num_segments
    parts{i} = tses(:, begins(i) : begins(i) + segm_size - 1);
end

end