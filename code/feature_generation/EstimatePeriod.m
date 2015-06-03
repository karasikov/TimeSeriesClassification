function period = FindPeriod(ts)

g1 = ts(1 : end - 1);
g2 = ts(2 : end);

periods = [];
intersections = 0;
begin = 0;

for k = 1 : length(g1) - 1
    if (g1(k) / g2(k) - 1) * (g1(k + 1) / g2(k + 1) - 1) < 0
        intersections = intersections + 1;
    end
    if intersections > 2
        periods = [periods, k - begin];
        intersections = 0;
        begin = k;
    end
end

period = mean(periods);

end