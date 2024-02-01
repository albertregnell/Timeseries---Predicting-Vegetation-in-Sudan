%% Model of vegetation(ndvi) WITH rain as input 
% Time Series Analysis
% Niels Gudde & Albert Regnell

clear; clc;
close all

%% Loading data
[rain_org, rain_org_m1, rain_org_m2, rain_org_v, rain_org_t, ndvi_scaled, ndvi_m, ndvi_v, ndvi_t] = getDatasets();

a1 = 1;
[rain_m1, ~] = rain_predict(a1, rain_org_m1, 1);


%% Plot the datasets

t1 = linspace(datenum(1982 ,1 ,1) ,datenum(1999 ,12 ,31) , 480-264);
t2 = linspace(datenum(1982 ,1 ,1) ,datenum(1999 ,12 ,31) , length(ndvi_scaled));

% We now plot the vegetation and the rain
subplot(211)
plot(t2, ndvi_scaled)
title("vegetation, ndvi")
datetick('x');

subplot(212)
plot(t1, rain_org(265:end))
title("rain")
datetick('x');

print -deps plotData

%% Check if we should transform the data

checkIfNormal(rain_m1, 'rain modeling data');
rain_m1_log = iddata(log(rain_m1+1))';
checkIfNormal(rain_m1_log.y, 'log(rain) modeling data');
% The log-transformed rain data looks more normal, we proceed with that

checkIfNormal(ndvi_m, 'NDVI modeling data');
ndvi_m_log = log(ndvi_m);
checkIfNormal(log(ndvi_m), 'log(NDVI) modeling data');
% The log-transformed vegetation data is NORMAL, we proceed with that


%% Lets try to instead simulate data according to a BJ process and see if our code 
% identifies the crosscorrelation
% 
% rng(0)
% n = length(ndvi_m_log); % Number of samples
% extraN = 120;
%  
% [x, ~] = simulateRainData(0.80, n+extraN); % Create the input
% 
% A2 = [1 0.90];
% B = [0.9 -0.8, 0.9, 0.8, 0.9];
% 
% C=1;
% A1 = [1 -.65];
% e = sqrt(1.5) * randn(n + extraN,1);
% 
% y = filter(C,A1,e) + filter(B,A2,x); % Create the output
% x = x(extraN+1:end) ; y = y(extraN+1:end); % Omit initial samples
% 
% rain_m1_log = iddata(x);
% ndvi_m_log = y;
% 
% %clear A1 A2 C B e w A3 C3 x y % Clear all variables except for the simulated data
% 
% subplot 211
% plot(ndvi_m_log)
% subplot 212
% plot(rain_m1_log.y)

%% First we want to model the rain as an ARMA process
plotACFnPACFnNoArm(rain_m1_log.y, 100, 'log(rain) modeling data');

%% We see a have a strong period in 36. Add those as AR and MA process.
% 
A = [1 zeros(1, 36)];
C = [1 zeros(1, 36)];

model_init = idpoly(A, [], C);
model_init.Structure.a.Free = [0 0 0 1 zeros(1, 5) 1 zeros(1, 26) 1];
model_init.Structure.c.Free = [0 1 1 zeros(1, 33) 1];
model_sarima = pem(rain_m1_log, model_init);

w_hat = resid(model_sarima, rain_m1_log);
w_hat = myFilter(w_hat, length(A)-1);

plotACFnPACFnNorm(w_hat.y, 50, 'log(rain), SARIMA(0, 37, 0, 36)', 0);
figure;
whitenessTest(w_hat.y);
present(model_sarima);
figure;
plot(w_hat.y)

%This is white! 



%% We now move on to model ndvi using the rain
% We want to model ndvi as an ARMAX, we use the Box Jenkings model since it
% is more general. 

% We start by prewhitening log(ndvi)
eps = filter(model_sarima.A, model_sarima.C, ndvi_m_log);
eps = myFilter(eps, length(model_sarima.A)-1);
figure;
plot(eps)

%% Compute CCF from w to eps 
M=50; % Number of lags we want to see in the CCF
n = length(ndvi_m_log);
[what_temp,eps_temp] = sameLength(w_hat.y,eps);
figure;

stem(- M : M, crosscorr(what_temp, eps_temp, M)); 
title('Cross correlation function'), 
xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1),'--') 
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1),'--') 
hold off

% We look at CCF and should be able to estimate the impulse response. It is
% always in between the confidence intervals and we probably cannot trust
% it.

%% Plotting just to check if we see correlation
plot(what_temp);
hold on
plot(eps_temp);
legend('what', 'eps');

%% cross x to y just to check

figure; 
[rain_temp,ndvi_temp] = sameLength(rain_m1_log.y,ndvi_m_log);
stem(- M : M, crosscorr(rain_temp, ndvi_temp, M)); 
title('Cross correlation function'), 
xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1),'--') 
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1),'--') 
hold off

%correlated as fuck. Does all correlation really disappear just because we
%pre-whiten?

%% plotting y and x to see correlation
figure;
plot(ndvi_temp);
hold on;
plot(rain_temp);
legend('log vegetation', 'log rain');


%% Decide transfer function H
% We look at CCF and should be able to estimate the impulse response
% We can't trust the cross correlation so we try with the simplest version,
% i.e only b0


A2 = [1];
B = [0];
Mi = idpoly (1 ,B ,[] ,[] ,A2);
Mi.Structure.b.Free = [1];
[rain_log_temp, ndvi_log_temp] = sameLength(rain_m1_log.y, ndvi_m_log);
z = iddata(ndvi_log_temp,rain_log_temp); % Why is y in here?

% We create an estimatee of H ((B*z^(-d))/A2) part in 4.50. This is the
% transfer function from x to y (same as from w to eps)
Mba2 = pem(z,Mi); 
present(Mba2)

% We look at the residuals, e_tilde, which later will be modeled as (C1/A1)*e
% where e is white.
etilde = resid(Mba2, z); 
etilde = myFilter(etilde, length(Mi.B)-1);

%% We want etilde and rain to be uncorrelated, lets look at the correlation
figure; 
[rain_temp,etilde_temp] = sameLength(rain_m1_log.y,etilde.y); %vilket håll ska man kolla?
stem(- M : M, crosscorr(etilde_temp ,rain_temp, M)); 
title('Cross correlation function'), 
xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1),'--') 
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1),'--') 
hold off

%% We now check if e_tilde is white

plotACFnPACFnNorm(etilde.y, 50, 'Etilde residuals');
figure;
whitenessTest(etilde.y)
figure;
plot(etilde.y);
%checkIfNormal(log(etilde.y+1-min(etilde.y)), 'etilde') %Maybe check
%normality
 
%it's not white. It has to be modeled as an ARMA process 
% (as e_tilde = (C1/A1)*e)

%% We suspect there should be a season of 36, and the ACF supports this

S = 36;
AS = [1 zeros(1, S-1) -1];
model_s = idpoly(AS);

etilde_s = resid(model_s, ndvi_m);
etilde_s = myFilter(etilde_s, length(AS));

plotACFnPACFnNorm(etilde_s.y, 50, 'nabla_{36} ndvi');
present(model_s);
figure;
whitenessTest(etilde_s.y);
%% We see a strong PACF at lag 1, lets remove it. 

A1 = [1 0];
C1 = [1 zeros(1,36)];

model_init = idpoly(A1, [], C1);
model_init.Structure.a.Free = [0 1];
model_init.Structure.c.Free = [0 zeros(1,35) 1];
model2 = pem(etilde_s.y, model_init);

e2 = resid(model2, etilde_s.y);
e2 = myFilter(e2, length(model2.A)-1);

% Presenting the model
present(model2);
plotACFnPACFnNorm(e2.y, 38, "AR(1) model etilde", 0);
figure;
whitenessTest(e2.y);
figure;
plot(e2.y)
% It is white!


%% Checking if all dependence from e_tilde was removed in x

[rain_temp,e2_temp] = sameLength(rain_m1_log.y,e2.y);
figure;
stem(- M : M, crosscorr(e2_temp ,rain_temp, M));
title('Cross correlation function'), 
xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1),'--') 
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1),'--') 
hold off

%looks like it was removed

%% Lets estimate all the polynomials together

A1_temp = conv(A1, AS); %Falta med diff 36? Varför -1 på näst sista
%A2= [1 0];
%B= [0 0 0 0 0];
C= [1];
%we don't need C3 and A3 anymore

Mj = idpoly (1 ,B,C,A1_temp,A2); 
Mj.Structure.b.Free = [1];
Mj.Structure.d.Free = [0 1 zeros(1, 34) 1 1];
%Mj.Structure.c.Free = [0 zeros(1, 35) 1];

[rain_log_temp, ndvi_log_temp] = sameLength(rain_m1_log.y, ndvi_m_log);

z = iddata(ndvi_log_temp,rain_log_temp);
MboxJ = pem(z ,Mj); 
present (MboxJ)
ehat = resid(MboxJ,z);
ehat = myFilter(ehat, length(B)-1);

plotACFnPACFnNorm(ehat.y, 40, "ARMA model etilde", 0);
figure;
whitenessTest(ehat.y);

%Is white
%Parameters is not significant
% it was better without ma 36
% parameter uppskattningen är väldigt konstiga. B är typ 0 allihopa och A1
% är 



%% Want to check the crosscorrelation of the new residuals with x again

[ehat_temp, rain_log_temp] = sameLength(ehat.y, rain_m1_log.y);
figure;
stem(- M : M, crosscorr(ehat.y ,rain_log_temp, M));
title('Cross correlation function'), 
xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1),'--') 
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1),'--') 
hold off

%x and residuals are correlated but we do not trust it

%% Final model polynomials

A1 = MboxJ.D; 
A2 = MboxJ.F;
B = MboxJ.B;
C1 = MboxJ.C;

% Note that Matlab uses a slightly different notation for the BJ model as
% compared to that used in the course. In Matlab's notation:
%
%   A(z) y(t) = [B(z)/F(z)] u(t) + [C(z)/D(z)] e(t)
%
% This means that:
%   A(z) = 1,       B(z) = B(z),    F(z) = A2(z)
%   C(z) = C1(z),   D(z) = A1(z)
