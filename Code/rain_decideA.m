%% Runs the reconstructingRainData function to find the best value of a1

% Loading the data
clear; clc;
close all;

load proj23.mat
x_bad = ElGeneina.rain;
y = ElGeneina.rain_org;

n = 200; % Number of different a1 to test

% Initializing the vectors in which to save the performance
res_model_vec = zeros(1, n+1);
sumDiff_model_vec = zeros(1, n+1);
res_fixNeg_vec = zeros(1, n+1);
sumDiff_fixNeg_vec = zeros(1, n+1);
zero_line = zeros(1, n+1);

%% Running the reconstructingRainData function n times

for i = 0:n
    a = -1+2.2*i/n;
    [res_model_vec(i+1), sumDiff_model_vec(i+1), res_fixNeg_vec(i+1), sumDiff_fixNeg_vec(i+1), ~] = rain_reconstruct(a,y);
end

a1 = linspace(-1, a, n+1);

figure;
subplot 211
hold on
plot(a1, res_model_vec)
plot(a1, res_fixNeg_vec)
hold off
legend('res model', 'res model fixed negative')
subplot 212
hold on
plot(a1, sumDiff_model_vec)
plot(a1, sumDiff_fixNeg_vec)
plot(a1, zero_line)
hold off
legend('sum diff model', 'sum diff model fixed negative')


%% Checking with simulated data

[x_sim, y_sim] = rain_simulate(0.7, 1440);

a = 1;
[res_model_vec(i+1), sumDiff_model_vec(i+1), res_fixNeg_vec(i+1), sumDiff_fixNeg_vec(i+1), rain_est] = rain_reconstruct(a, y_sim);

figure;
hold on
plot(x_sim)
plot(y_sim)
plot(rain_est)
hold off
legend('true rain','y', 'reconstructed rain')

% It looks pretty good.
