% On Preserving Statistical Characteristics of Accelerometry
% Data using their Empirical Cumulative Distribution

% calculate ECDF at n points
function X = ECDF_interpolated(ts, n)
    [f, x] = ecdf(ts + randn(size(ts))*0.01*std(ts));
    X = interp1(f, x, linspace(0, 1, n), 'PCHIP');
end