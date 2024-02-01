% Simulates rain data over N periods with a1 coefficient a1
% returns the data as well as the sum of every three periods as vectors

function [x, y] = rain_simulate(a1, N)

extraN = 100;
e = randn(N+extraN, 1);

x = filter(1, [1 -a1], e);
x = myFilter(x, extraN);

for i = 1:length(x)
    if x(i)<0
        x(i) = 0;
    end
end

y = zeros(N/3, 1);

for i = 0:length(y)-1
    y(i+1) = x(3*i+1) + x(3*i+2) + x(3*i+3);
end

    