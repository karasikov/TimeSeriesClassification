function [ w, y_est ] = CalcArGarch( x, ar_len, arch_len, garch_len )

X = zeros(length(x) - ar_len, ar_len);
y = zeros(length(x) - ar_len, 1);

for idx = ar_len + 1 : length(x)
    X(idx - ar_len, :) = x(idx - ar_len : idx - 1);
    y(idx - ar_len) = x(idx);
end

X = [ones(length(y), 1), X];
w = (X' * X) ^ (-1) * (X' * y);
y_est = X * w;

vars = ((x(ar_len + 1 : end))' - y_est) .^ 2;
X_sq = (X(garch_len + 1 : end, :)) .^ 2;
X_sq = [X_sq(:, 1), X_sq(:, end - arch_len + 1 : end)];
X_grch = zeros(length(vars) - garch_len, garch_len);
y_vars = zeros(length(vars) - garch_len, 1);
for idx = garch_len + 1 : length(vars)
    X_grch(idx - garch_len, :) = (vars(idx - garch_len : idx - 1))';
    y_vars(idx - garch_len) = vars(idx);
end
X_sq = [X_sq, X_grch];
arch_w = (X_sq' * X_sq) ^ (-1) * X_sq' * y_vars;

w = [w; arch_w];

end

