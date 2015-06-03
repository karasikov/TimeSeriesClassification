% split by segments of size |segm_size|
%
function [parts] = partition(tses, segm_size)

n = size(tses, 2);
if n < segm_size
    segments_length = [n];
else
    segments_length = repmat(segm_size, 1, floor(n / segm_size));
end

parts = mat2cell(tses(:, 1 : sum(segments_length)), ...
                 size(tses, 1), ...
                 segments_length);

end