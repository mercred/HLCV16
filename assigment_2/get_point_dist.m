%
% computes all distances of potential point correspondances between 
% the images. Uses the descriptors given as input. Computes the distances
% by applying the chi_square metric.
% 
% D1   : set of descriptors computed from interest points in image 1
% D2   : set of descriptors computed from interest points in image 2
%
% D    : Distance matrix, each entry corresponds to a possible pair
%
function D = get_point_dist(D1,D2)

sizeD1 = size(D1,1);
sizeD2 = size(D2,1);
D = zeros(sizeD1, sizeD2);

% loop over all possible pairs and compute a score
for i = 1:sizeD1
    for j = 1:sizeD2
        d = dist_chi2(D1(i,:),D2(j,:));
        sd = sum(d);
        D(i,j) = sd;
    end
end
        