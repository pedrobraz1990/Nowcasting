actuals = csvread('actuals.csv');
forecasts = csvread('forecasts.csv');

[out1, out2] = bounds_tests_function_40nan_simple(actuals,forecasts,1)

csvwrite('out.csv',out1)