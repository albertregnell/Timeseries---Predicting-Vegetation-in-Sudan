% Reconstructs rain data using the given data y, 
% the interpolated data (divided by 3) and the a1 

% Returns 
% res               = the difference in total rain between the reconstructed data and y
% sumDiff           = sum of squared res
% res_fixNeg        = difference in total rain when handling negative values
% sumDiff_fixNeg    = sum of squared res when handling negative values
% rain_est          = the reconstructed rain

function [res, sumDiff, res_fixNeg, sumDiff_fixNeg, rain_est] = rain_reconstruct(a1, y)

m = 0;
y = y-m;

% Estimate the unknown parameters using a Kalman filter

p0 = 3; % Number of unknown x
N = length(y)*3;

A = [a1 0 0; 1 0 0; 0 1 0];

Rw    = 0.1;                                      
Re    = [0.04 0 0; 0 1e-7 0; 0 0 1e-7]; 
Rx_t1 = 10*eye(p0); 
h_et  = zeros(N,1);                          % Estimated one-step prediction error.
xt    = zeros(p0,N);                         % Estimated states. 
xt(:, 3) = [y(1)/3; y(1)/3; y(1)/3];         % Intial state
yhat  = zeros(N,1);                          % Estimated output.
xStd  = zeros(p0,N);                         % Stores one std for the one-step prediction.


for t=4:N                                      
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    C    = [1 1 1];     
    
    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    yhat(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
    
    % Update ehat with the true y if we have it
    if mod(t, 3) == 0
        h_et(t) = y(t/3)-yhat(t);
    else
        h_et(t) = 0;
    end

    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Estimate a one std confidence interval of the estimated parameters.
    % This is only for the plots.
    xStd(:,t) = sqrt( diag(Rx_t) );            % This is one std for each of the parameters for the one-step prediction.
end


% plotting the estimates
rain_est = xt(1, :)+m/3;
%rain_est = xt(1, :)
for i = 1:length(rain_est)-1
    if rain_est(i)<0
        %rain_est(i+1) = rain_est(i+1) + rain_est(i);
        rain_est(i) = 0;
    end
end


% Ways of validating

%now we calculate h_et and squared residualsbased on the final x states
%final_rain = xt(1,:);
final_rain = xt(1,:)+m/3;
reconstructed_y1 = zeros(1, length(y));
for i = 1:length(y)
    reconstructed_y1(i) = sum(final_rain((3*i - 2):(3*i)));
end

%e_hat2 = reconstructed_y1'-y
e_hat2 = reconstructed_y1'-(y+m);
res = sum(e_hat2.^2);
sumDiff = sum(y)-sum(reconstructed_y1);

%This is the efter fixing the negative rain
reconstructed_y2 = zeros(1, length(y));
for i = 1:length(y)
    reconstructed_y2(i) = sum(rain_est((3*i - 2):(3*i)));
end

e_hat3 = reconstructed_y2'-(y+m);
res_fixNeg = sum(e_hat3.^2);
sumDiff_fixNeg = sum(y)-sum(reconstructed_y2);

