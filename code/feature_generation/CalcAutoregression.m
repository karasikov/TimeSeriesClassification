function [ w, y_est ] = CalcAutoregression( x, ar_len )

X = zeros(length(x) - ar_len, ar_len);
y = zeros(length(x) - ar_len, 1);

for idx = ar_len + 1 : length(x)
    X(idx - ar_len, :) = x(idx - ar_len : idx - 1);
    y(idx - ar_len) = x(idx);
end

X = [ones(length(y), 1), X];
w = (X' * X) ^ (-1) * (X' * y);
y_est = X * w;

end
