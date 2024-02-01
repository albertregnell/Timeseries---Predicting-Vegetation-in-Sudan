
%% Model of vegetation(NVDI) WITHOUT using rain as input 
% Time Series Analysis
% Niels Gudde & Albert Regnell

clear; clc;
close all

%% Loading data
load proj23.mat

nvdi = ElGeneina.nvdi;

% We rescale the vegetation data to the range -1 to 1
nvdi_scaled = (nvdi-127.5)/127.5;

% We plot the vegetation
plot(nvdi_scaled)
title("vegetation, nvdi")

%% Dividing the datasets 
% We need to divide our data into modeling, validation, test

percent_m = 0.70;
percent_v = 0.20;
percent_t = 0.10;

m_end = floor(percent_m*length(nvdi));
v_end = floor((percent_m+percent_v)*length(nvdi));


nvdi_m = nvdi_scaled(1:m_end); 
nvdi_v = nvdi_scaled(m_end+1:v_end);
nvdi_t = nvdi_scaled(v_end+1:end);

%% Check if we should transform the data

checkIfNormal(nvdi_m, 'NDVI modeling data')

bcNormPlot(nvdi_m) % Peak in 0.22, we choose to log-transform the data

nvdi_m_log = log(nvdi_m+2);
checkIfNormal(nvdi_m_log, 'log(NDVI) modeling data') % Normal!

%% Simulate

% n = 10000;
% e = randn(n+100,1); %white noise
% nvdi_m = filter(C_final,A_final,e); % Create the output
% nvdi_m_log = log(nvdi_m+2)
% nvdi_m_log = myFilter(nvdi_m_log,100);
% figure;
% plot(nvdi_m_log);


%% We see a strong spike in the PACF at lag 1, lets remove it

% This is to create a model wothout differentiating first

% 
% poly_init = idpoly([1 0] ,[] ,[1 zeros(1, 36)]);
% poly_init.Structure.a.Free = [0 1];
% poly_init.Structure.c.Free = [zeros(1,36) 1];
% model_armax = pem(nvdi_scaled, poly_init);
% 
% res = resid(model_armax, nvdi_scaled);
% res = myFilter(res, length(model_armax.C)-1);
% 
% % Evaluate the model
% plotACFnPACFnNorm(res.y, 50, 'AR(1) residuals');
% figure;
% whitenessTest(res.y);
% present(model_armax);

%% We suspect there should be a season of 36, and the ACF supports this

S = 36;
AS = [1 zeros(1, S-1) -1];
model_s = idpoly(AS);

nvdi_s = resid(model_s, nvdi_m_log);
nvdi_s = myFilter(nvdi_s, length(AS));

plotACFnPACFnNorm(nvdi_s.y, 50, 'nabla_{36} nvdi');
present(model_s);
figure;
whitenessTest(nvdi_s.y);

%% We see a strong peak in the PACF at lag 1, lets remove it
A = [1 0];
C = [1];

model_init = idpoly(A, [], C);
%model_init.Structure.a.Free = [0 1 zeros(1, 34) 1];
%model_init.Structure.c.Free = [zeros(1, 36) 1];
model_sarima = pem(nvdi_s, model_init);

res = resid(model_sarima, nvdi_s);
res = myFilter(res, length(C)-1);

plotACFnPACFnNorm(res.y, 50, 'NVDI, SARIMA(36, 1, 0, 0)');
figure;
whitenessTest(res.y);
present(model_sarima);

%% We now have peaks at lag 36 in both the ACF and the PACF

A = [1 0];
C = [1 zeros(1, 36)];

model_init = idpoly(A, [], C);
%model_init.Structure.a.Free = [0 1 zeros(1, 34) 1];
model_init.Structure.c.Free = [zeros(1, 36) 1];
model_sarima = pem(nvdi_s, model_init);

res = resid(model_sarima, nvdi_s);
res = myFilter(res, length(C)-1);

plotACFnPACFnNorm(res.y, 50, 'NVDI, SARIMA(36, 1, 0, 36)');
figure;
whitenessTest(res.y);
present(model_sarima);

%We conclude that this is the is a good enough model with white residuals.
%This is an SARIMA(36,1,0,36)

%% Final parameter values are:
A_simulate = conv(model_sarima.A, AS); % Perform polynomial multiplication using convolutions
C_simulate= model_sarima.C;

%A_simulate = [1 -0.7837 zeros -1 0.7837]
%C_simulate = [1]
%exactly the same


%% Lets check how the model performes on the validation dataset

%% Lets predict using the SARIMA model
A_final = conv(model_sarima.A, AS); % Perform polynomial multiplication using convolutions
C_final= model_sarima.C;
modelLim = length(nvdi_m);
n = length(nvdi_m) + length(nvdi_v);

%A_final = [1 -0.7786 zeros -1 0.7786]
%C_final = [1]

k = 7; 
% Form the SARIMA prediction for y_t
[Fy, Gy] = polydiv(C_final, A_final, k );

% Form the predicted y
yhatk  = filter(Gy, C_final, [nvdi_m', nvdi_v']');

% Plotting the predicted NVDI vs the actual
figure
plot([[exp(nvdi_m')-1, exp(nvdi_v')-1]' exp(yhatk)-1] ) % Transform the data back
line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' ) %modelLim gives error for me
legend('Output signal', 'Predicted output', 'Prediction starts')
title( sprintf('Predicted output signal, y_{t+%i|t}', k) )
axis([1 n -1 1]);

% We look at the prediction error residuals
ehat_sarima = nvdi_v-yhatk(length(nvdi_m)+1:end); %Let it start on corrupted data???
pred_var_sarima = var(ehat_sarima); 


%% Compare to the naive predictor

y_naive = naivePred([nvdi_m', nvdi_v']', k)';
ehat_naive = nvdi_v-y_naive(length(nvdi_m)+1:end);

pred_var_naive = var(ehat_naive);

figure
plot([[nvdi_m', nvdi_v']' y_naive] )
line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' ) %modelLim gives error for me
legend('Output signal', 'Predicted output', 'Prediction starts')
title( sprintf('Predicted output signal, y_{t+%i|t}', k) )
axis([1 n 0 max(nvdi_m)*1.5])

% The SARIMA model is better than the naive predictor both for 
% k = 3 and k = 39. Same prediction error variance for k = 36.


%% Compare to the naive predictor

y_naive = naivePred([nvdi_m', nvdi_v']', k)';
ehat_naive = nvdi_v-y_naive(length(nvdi_m)+1:end);

pred_var_naive = var(ehat_naive);

figure
plot([[nvdi_m', nvdi_v']' y_naive] )
line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' ) %modelLim gives error for me
legend('Output signal', 'Predicted output', 'Prediction starts')
title( sprintf('Predicted output signal, y_{t+%i|t}', k) )
axis([1 n 0 max(nvdi_m)*1.5])

% The SARIMA model is better than the naive predictor both for 
% k = 3 and k = 39. Same prediction error variance for k = 36.




