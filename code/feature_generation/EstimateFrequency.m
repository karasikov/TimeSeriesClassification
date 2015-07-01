function [f] = EstimateFrequency(ts)

[w,~] = model_fourier((1:length(ts))', ts', 2);
f = w(end);

end