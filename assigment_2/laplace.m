%
% compute Laplacian of the image
%
% note: use the function gaussderiv2.m
%

function imgLap = laplace(img,sigma)
    [imgDxx, imgDxy, imgDyy] = gaussderiv2(img,sigma);
    imgLap = imgDxx + imgDyy;
  % ... 

