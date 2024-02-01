% Returns a new dataset with naive predictions of the given dataset
% Naive prediction: 
% We think the value k steps ahead will be the same as the current value

function y_naive = naivePred(ytrue, k, sameAsCurrent)


if sameAsCurrent == 1
    n = length(ytrue);
    y_naive = zeros(1, n);
    for i = k+1:n
        y_naive(i) = ytrue(i-k);
    end
else
    n = length(ytrue);
    y_naive = zeros(1, n);

    m = max(k, 36);
    for i = m+1:n
        y_naive(i) = ytrue(i-36);
    end
end


