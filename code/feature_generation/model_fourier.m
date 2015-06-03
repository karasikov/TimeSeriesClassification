function [w, gof] = model_fourier(t, x, level)
% Function returns model coefficients for Fourier Series
%
% input:
%   x - vector [1,n]
%   level - order in {1,...,8}
%
% output:
%   w - vector of model parameters
%   gof - goodnes-of-fit vector:
%       [sse, rsquare, dfe, adjrsquare, rmse]
%
[f, gof] = fit(t, x, strcat('fourier', num2str(level)));

w = coeffvalues(f);
gof = cell2mat(struct2cell(gof))';

end