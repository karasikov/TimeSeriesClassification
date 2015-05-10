function [ params ] = CalcEigenvalues( ts, ar_len )

% Initialize features
params = zeros(1, (ar_len + 1));

x = zeros(length(ts) - ar_len, ar_len);
for idx = ar_len + 1 : length(ts)
    x(idx - ar_len, :) = ts(idx - ar_len : idx - 1);
end

x = [ones(size(x, 1), 1), x];
[~,v,~] = svd(x' * x);
params(1 : ar_len + 1) = (diag(v))';

params = sqrt(params);

end