%
% return 2nd order Gaussian derivatives of the image
% 
% note: use functions gauss.m and gaussdx.m from exercise 1
%

function [imgDxx, imgDxy, imgDyy] = gaussderiv2(img,sigma)
  
  assert(length(size(img)) == 2, 'expecting 2d grayscale image');
  
  [Fx, x] = gaussdx(sigma);
  [F, x] = gauss(sigma);
  %{
  imgDx  = conv2(img,   F,  'same');
  imgDxx = conv2(imgDx, F,  'same');
  imgDy  = conv2(img,   F', 'same');
  imgDyy = conv2(imgDy, F', 'same');
  imgDxy = conv2(imgDx, F', 'same');
  %}
  %imgDy = conv2()
  
  
  %Fdxx = (-sigma^2)*(F') * (F + Fx .* x);
  %{
  imgDxx = conv2(img, Fdxx,  'same');
  imgDyy = conv2(img, Fdxx', 'same');
  
  Fdxy = (Fx') * Fx;
  imgDxy = conv2(img, Fdxy, 'same');
  %}
 
  
  [imgDx,imgDy]=gaussderiv(img,sigma);
  [imgDxx,imgDxy]=gaussderiv(imgDx,sigma);
  [aaa,imgDyy]=gaussderiv(imgDy,sigma);
  
  % ... 



