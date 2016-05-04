% 
% compute euclidean distance between x and y
% 


function d = dist_l2(x,y)
    d = x-y;
    d = d.*d;
    d = sum(sum(sum(d)));
  % ...
