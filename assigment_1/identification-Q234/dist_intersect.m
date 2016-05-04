% 
% compute intersection distance between x and y
% return 1 - intersection, so that smaller values also correspond to more similart histograms
% 

function d = dist_intersect(x, y)
    d = min(x,y);
    d = sum(sum(sum(d)));
    d = 1 - d;
  % ...