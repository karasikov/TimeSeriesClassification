# Feature-based time-series classification

Here, we consider a time series as a sequence of its segments approximated by parametric models (e.g. autoregressive model, discrete Fourier transform, discrete wavelet transform).
The parameters of the approximating models are used as time-series' features.  
Then, we generalize this approach and use the distributions of the parameters estimated for models approximating different time-series' segments.  

The proposed approach is applied to the problem of human activity recognition from accelerometer data.

## Reference
> M. E. Karasikov, V. V. Strijov, Feature-based time-series classification, Inform.
Primen., 2016, Volume 10, Issue 4, 121–131.  
URL: http://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=ia&paperid=452&option_lang=eng  
DOI: [10.14357/19922264160413](https://doi.org/10.14357/19922264160413)  
pdf: [full text](http://www.mathnet.ru/php/getFT.phtml?jrnid=ia&paperid=452&what=fullt&option_lang=eng)

### Matlab code
Matlab code resides in [code](./code).

### Python code
All code rewritten in Python for reproducing experiments from the paper can be found in [code/python](./code/python).

### Interactive human activity recognition application
An interactive online application is running on http://www.karasikov.com/activity.
The source code resides in [code/activity_prediction](./code/activity_prediction).

#### Server
The full code of the demonstration server can be found in
[code/activity_prediction/server](./code/activity_prediction/server).

#### Android client application
To get the full code of the android application, unpack ActivityPrediction.rar in [code/activity_prediction/android](./code/activity_prediction/android).
