function distribution_parameters = feature_generation(random_sample)

distribution_parameters = [ mean(random_sample, 1), var(random_sample, 0, 1) ];

end