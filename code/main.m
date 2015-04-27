load('../data/USC-HAD/USC_HAD_dataset.mat')

ts_lengths = cellfun(@(sample)( size(sample.ts,1) ), dataset);
mean_ts_length = mean(ts_lengths);

Mdl = arima(100,1,2)
Mdl = estimate(Mdl, dataset{1}.ts(1:1000,4))
y = Mdl.forecast(1000)
plot([dataset{1}.ts(1:1000,4); y])