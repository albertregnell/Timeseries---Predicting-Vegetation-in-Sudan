% Reconstructs rain data using the given data y, 
% the interpolated data (divided by 3) and the a1 

% Returns 
% res               = the difference in total rain between the reconstructed data and y
% sumDiff           = sum of squared res
% res_fixNeg        = difference in total rain when handling negative values
% sumDiff_fixNeg    = sum of squared res when handling negative values
% rain_est          = the reconstructed rain

function [rain_est, rain_pred] = rain_predict(a1, y, k)

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
yhatk  = zeros(N,1);                         % k step prediction for y
xhatk = zeros(N, 1);                         % k step prediction for y
yhat1  = zeros(N,1);                         % Stores one step predictions
xStd  = zeros(p0,N);                         % Stores one std for the one-step prediction.


for t=4:N-k                                     
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

    % Form the k-step prediction by first constructing the future C vector
    % and the one-step prediction. Note that this is not yhat(t) above, as
    % this is \hat{y}_{t|t-1}.
    Ck = [1 1 1];           % C_{t+1|t}
    yk = Ck*xt(:,t);                  % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}
    yhat1(t+1) = yk; %saving the 1 step predictions

    % Note that the k-step predictions is formed using the k-1, k-2, ...
    % predictions, with the predicted future noises being set to zero. If
    % the ARMA has a higher order AR part, one needs to keep track of each
    % of the earlier predicted values.
    Rx_k = Rx_t1;
    for k0=2:k
        Ck = [1 1 1];                           % C_{t+k|t}
        yk = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        Rx_k = A*Rx_k*A' + Re;                  % R_{t+k+1|t}^{x,x} = A R_{t+k|t}^{x,x} A^T + Re  
    end
    xhatk(t+k) = xt(1,t);                       % Best predictions is todays value?
    yhatk(t+k) = yk;                            % Note that this should be stored at t+k.

    % Estimate a one std confidence interval of the estimated parameters.
    % This is only for the plots.
    xStd(:,t) = sqrt( diag(Rx_t) );            % This is one std for each of the parameters for the one-step prediction.
end


% plotting the estimates
rain_est = xt(1, :)'+m/3;
rain_pred = xhatk;
%rain_est = xt(1, :)
for i = 1:length(rain_est)-1
    if rain_est(i)<0
        %rain_est(i+1) = rain_est(i+1) + rain_est(i);
        rain_est(i) = 0;
    end
    if rain_pred(i)<0
        rain_pred(i) = 0;
    end
end

% Validating the predictions

% Checking predicted x values



