%% Time to predict rain and evaluate model
clear; clc;

k = 7;

%% Loading data
[rain_org, rain_org_m1, rain_org_m2, rain_org_v, rain_org_t, ndvi_scaled, ndvi_m, ndvi_v, ndvi_t] = getDatasets();

a = 1;
[rain_m1, ~] = rain_predict(a, rain_org_m1, 1);

% Final model polynomials (from modelInput)
A1 = [1.0000 -0.8002 zeros(1, 34) -0.217 0.02147]; %MboxJ.D
C1 = 1;         %MboxJ.C

A2 = 1;         %MboxJ.F
B = 0.0211;     %MboxJ.B


%% Predict rain as input

[rain_v, rain_pred] = rain_predict(a, [rain_org_m1; rain_org_v], k);
figure;
hold on;
plot(rain_v);
plot(rain_pred);
legend('rain reconstructed', 'rain predicted');
hold off;
title( sprintf('Predicted rain, x_{t+%i|t}', k) )


%% We have log rain as input
rain_log = log(rain_v+1)';
figure;
plot(rain_log);


%%

%Gk is obtained from Deiophantine equation:
[Fk, Gk] = polydiv( C, A, k );

%To make a k-step prediction one use
yhat_k = filter( Gk, C, svedala);

% We have to remove the initial samples
yhat_k = myFilter(yhat_k, length(C));
[svedala_temp, yhat_k_temp] = sameLength(svedala, yhat_k);

% Thus we can calculate the variance of the noise as
ehat1 = svedala_temp-yhat_k_temp;
noise_var = var(ehat1); % The variance is 0.3754



%% Examine the k-step prediction using k = 3 and 26
k = 26;

%Gk is obtained from Diophantine equation:
[Fk, Gk] = polydiv( C, A, k );

%To make a k-step prediction one uses
yhat_k = filter( Gk, C, svedala);

% We have to remove the initial samples
yhat_k = myFilter(yhat_k, length(C));
[svedala_temp, yhat_k_temp] = sameLength(svedala, yhat_k);

% Thus we can calculate the prediction error as
ehat = svedala_temp-yhat_k_temp;

%% Plot the temperature and the prediction
figure;
plot(yhat_k_temp)
hold on;
plot(svedala)


%% Calculate the mean prediction error
mean(ehat)
% k = 3 gives mean -0.011
% k = 26 gives mean -0.0086

%% Calculate prediction variance
% Theoretical
pred_var_theo = power(norm(Fk), 2)*noise_var
% k = 3 gives theoretical prediction variance 2.7493
% k = 26 gives theoretical prediction variance 12.6262

% Estimated
pred_var_est = var(ehat)
% k = 3 gives estimated prediction variance 2.6120
% k = 26 gives estimated prediction variance 10.7382


%% Calculate confidence intervals

quant95 = norminv(0.975)*sqrt(noise_var)*norm(Fk);

conf_int = [-quant95, quant95];

outside_int = sum(ehat > quant95) + sum(ehat < -quant95);
outside_percent = outside_int/length(ehat); 
% k = 3 gives that 6 percent of errors are outside conf int
% k = 26 gives that 3.3 percent of error are outside

%% 
figure
plot(ehat);
title('prediction error');

figure;
plot(covf(ehat, 40));
title('covariance prediction error');

plotACFnPACFnNorm(ehat, 40, 'mamma');

%With k = 3 it looks like an MA(2)
%if the white is really good it can look long in the future, more exposed
%and bigger error or something. We can ask about this
