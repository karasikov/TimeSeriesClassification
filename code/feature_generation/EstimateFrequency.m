function f = EstimateFrequency(ts)

sampleFreq = 1;
sampleTime = 1 / sampleFreq;
numSamples = length(ts);
timeVec    = (0 : numSamples - 1) * sampleTime;

transformedSignal = fft(ts');

powerSpectrum = transformedSignal .* conj(transformedSignal) / numSamples;
powerSpectrum = powerSpectrum(1 : numSamples / 2 + 1);

frequencyVector = sampleFreq / 2 * linspace( 0, 1, numSamples / 2 + 1 );

[~,idx] = max(powerSpectrum);
f = frequencyVector(idx);

end