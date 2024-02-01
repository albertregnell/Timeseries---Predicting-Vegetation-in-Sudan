%% Uses a Kalman filter to predict the vegetation

clear; clc;
close all;

k = 7;
N = 583;
a = 1;

[rain_org, rain_org_m1, rain_org_m2, rain_org_v, rain_org_t, ndvi_scaled, ndvi_m, ndvi_v, ndvi_t] = getDatasets();


%% Predict rain as input

[x, xhatk] = rain_predict(a, rain_org(1:end-22), k);
x = x(792:end); 
xhatk = xhatk(792:end);

figure;
hold on;
plot(x);
plot(xhatk);
legend('rain reconstructed', 'rain predicted');
hold off;
title( sprintf('Predicted rain, x_{t+%i|t}', k) )

%%
% Final model polynomials (from modelInput)
A1 = [1.0000 -0.8002 zeros(1, 34) -0.217 0.02147]; %MboxJ.D
C1 = 1;         %MboxJ.C

A2 = 1;         %MboxJ.F
B = 0.0211;     %MboxJ.B


%% Predict the output using the found model.
modelLim = 453;                                % Determine where the validation data starts.

y = [ndvi_m; ndvi_v];

KA = conv( A1, A2 );
KB = conv( A1, B );
KC = conv( A2, C1 );
[Fy, Gy]   = polydiv( C1, A1, k );
[Fhh, Ghh] = polydiv( conv(Fy, KB), KC, k );
yhatP = filter(Fhh, 1, xhatk) + filter(Ghh, KC, x) + filter(Gy, KC, y);
eP    = y(modelLim:end)-yhatP(modelLim:end);    % Form the prediction residuals for the validation data.

%% Now we construct a Kalman filter to estimate the parameters
% 
% For illustration purposes, we consider three different cases; in the first
% version, we estimate the parameters of the input; in the second, we 
% assume these to be fixed. In the third case, we modify the second case
% and examine if we can remove the KC parameter without losing too much
% performance. 
%
codeVersion = 1;
switch codeVersion
    case 1
        noPar   = 7;                            % The vector of unknowns is [ -KA(2) -KA(3) KB(1) KB(2) KB(3) KC(3) ]
        xt      = zeros(noPar,N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,2) = [ -KA(2) -KA(37) -KA(38) KB(1) KB(2) KB(37) KB(38)];
    case 2
        noPar   = 3;                            % The vector of unknowns is [ -KA(2) -KA(3) KC(3) ]
        xt      = zeros(noPar,N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,2) = [ -KA(2) -KA(37) -KA(38) ];
    case 3
        noPar   = 2;                            % The vector of unknowns is [ -KA(2) -KA(3)  ]
        xt      = zeros(noPar,N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,2) = [ -KA(2) -KA(37) -KA(38) ];
end

A     = eye(noPar);
Rw    = std(eP);                                % Measurement noise covariance matrix, R_w. Try using the noise estimate from the polynomial prediction.
Re    = 1e-6*eye(noPar);                        % System noise covariance matrix, R_e.
Rx_t1 = 1e-4*eye(noPar);                        % Initial covariance matrix, R_{1|0}^{x,x}
Rx_k  = Rx_t1;
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
yhatK = zeros(N,1);                             % Estimated output.
xStd  = zeros(noPar,N);                         % Stores one std for the one-step prediction.
startInd = 38;                                   % We use t-37, so start at t=38.
for t=startInd:N
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    switch codeVersion
        case 1                                  % Estimate all parameters.
            C = [ y(t-1) y(t-36) y(t-37) xhatk(t) x(t-1) x(t-36) x(t-37)];
            yhatK(t) = C*x_t1;
        case 2                                  % Note that KB does not vary in this case.
            C = [ y(t-1) y(t-2) h_et(t-2) ];
            yhatK(t) = C*x_t1 + KB * [xhatk(t) x(t-1) x(t-2)]';
        case 3
            C = [ y(t-1) y(t-2) ];              % Ignore one component.
            yhatK(t) = C*x_t1 + KB * [xhatk(t) x(t-1) x(t-2)]';
    end

    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    h_et(t) = y(t)-yhatK(t);                    % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rx_t) );             % This is one std for each of the parameters for the one-step prediction.
end

%%
plot(xt)



