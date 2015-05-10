% Histogram

function bins = BinsCount(ts, num_bins)
    edges = linspace(min(ts), max(ts), num_bins + 1);
    bins = histc(ts, edges) / length(ts);
    bins(end - 1) = bins(end - 1) + bins(end);
    bins(end) = [];
end
