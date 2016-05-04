% 
% compute chi2 distance between x and y
% 

function d = dist_chi2(x,y)
    d = x-y;
    s = x+y;
    s = s + (s==0); %get rid of zeros in denominator
    d = d.*d;
    d = d./s;
    d = sum(sum(sum(d)));

