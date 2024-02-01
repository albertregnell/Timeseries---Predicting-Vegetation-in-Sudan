%% Testing code


%% Testing the naive predictor
clear; clc;

y = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];
k = 5;

y_pred = naivePred(y, k);

plot(y)
hold on
plot(y_pred)

%% simulate AR(1)

n = 1000;
A = [1 1];
e = randn(n+100,1); %white noise
y = filter(1, A, e); % Create the output
y = myFilter(y,100);
figure;
plot(y);
plotACFnPACFnNorm(y, 20, 'AR(1)')