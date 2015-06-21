function [w, y_est] = Regress(y, X)

w = (X' * X) ^ (-1) * (X' * y);
y_est = X * w;

end