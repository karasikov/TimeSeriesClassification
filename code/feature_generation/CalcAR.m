function [w, y_est] = CalcAR(x, ar_len)

y = x(ar_len + 1 : end)';
X = zeros(length(x) - ar_len, ar_len);

for idx = ar_len + 1 : length(x)
    X(idx - ar_len, :) = x(idx - ar_len : idx - 1);
end

X = [ones(length(y), 1), X];

[ w, y_est ] = Regress(y, X);

end