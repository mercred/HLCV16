% px - vector of x coordinates of interest points
% py - vector oy y coordinates of interest points
% M - value of the cornerness function computed for every image pixel
%


function [px py M] = harris(img, sigma, thresh)
    alfa = 0.05;
    [G, x] = gauss(sigma * 1.5);  
    G = G*sigma;
    [imgDxx, imgDxy, imgDyy] = gaussderiv2(img,sigma);
   
    imgDxx = conv2(imgDxx, G, 'same');
    imgDxx = conv2(imgDxx, G', 'same');
    
    imgDyy = conv2(imgDyy, G, 'same');
    imgDyy = conv2(imgDyy, G', 'same');
    
    imgDxy = conv2(imgDxy, G, 'same');
    imgDxy = conv2(imgDxy, G', 'same');
    
    M = (imgDxx .* imgDyy - imgDxy.^2) - alfa * ((imgDxx + imgDyy).^2); 
    %M = M * sigma ^ 4;
    imgD = nonmaxsup2d(M);
    imgD = (imgD > thresh);
    [py,px] = find(imgD);
  % ... 
    