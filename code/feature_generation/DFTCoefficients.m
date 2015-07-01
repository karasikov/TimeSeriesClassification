function [w] = DFTCoefficients(ts, num_coefficients)
% Function returns fft coefficients
%   w - vector of model parameters

T = length(ts);
t = 1:T;
ffreq = 2 * pi / T;

F = fft(ts', T);
F = F(1 : round(T / 2));
omega = (0 : round(T / 2 - 1)) * ffreq; 
CA = 2 * real(F) / T;
CA(1) = CA(1) / 2;
CB = -2 * imag(F) / T;

% first coefficients for each

L = num_coefficients;
w = [CA(1:L+1)', CB(2:L+1)'];

% xapprox = CA(1)*ones(size(t)) ;
% for k=1:L
    % xapprox = xapprox + CA(k+1)*cos(omega(k+1)*t)...
    % + CB(k+1)*sin(omega(k+1)*t);
% end

% sst = sum((x-mean(x)).^2);
% sse = sum((x-xapprox).^2);
% rmse = sqrt(sse/size(x,1));
% rsquare = 1-sse/sst;
% gof = [sse,rsquare,rmse];

% subplot(2,2,2)
% plot(t,xapprox)

% power=sqrt(CA.^2 + CB.^2);
% subplot(2,2,3)
% plot(power)
% %axis([0 50 0 10])
% 
% subplot(2,2,4)
% plot(sqrt(CA.^2))
% axis([0 25 0 1])

end