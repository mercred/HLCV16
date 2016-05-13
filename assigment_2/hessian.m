% px - vector of x coordinates of interest points
% py - vector oy y coordinates of interest points
% H - value of Hessian determinant computed for every image pixel
%
% note: use the functions gaussderiv2.m and nonmaxsup2d.m 

function [px, py, H] = hessian(img, sigma, thresh)
    [imgDxx, imgDxy, imgDyy] = gaussderiv2(img,sigma);
    
    H = imgDxx .* imgDyy - (imgDxy).^2; 
    H = H * (sigma^4);
    
    imgD = nonmaxsup2d(H);
    imgD = (imgD > thresh);
    [py,px] = find(imgD > 0);
   % ...