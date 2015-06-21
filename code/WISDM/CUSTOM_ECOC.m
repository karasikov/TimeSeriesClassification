%% Classify <Downstairs> and <Upstairs> classes separately
function M = CUSTOM_ECOC()

M = OneVsAll(6);
M(:, 3:4) = [];
M = [M, [-1;-1;1;1;-1;-1]];
M = [M, [0;0;1;-1;0;0]];

end