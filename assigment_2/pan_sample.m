%
% This function stiches two images related by a homogrphy into one image. 
% The image plane of image 1 is extended to fit the additional points of
% image 2. Intensity values are looked up in image 2, using bilinear
% interpolation (use the provided function interpolate_2d). 
% Further parts belonging to image 1 and image 2 are smoothly blended. 
% 
%
% img1 : the first gray value image
% img2 : the second gray value image 
% H    : the homography estimated between the images
% sz   : the amount of pixel to increase the left image on the right 
% st   : amount of overlap between the images
%
% img  : the final panorama image
% 
function img = pan_sample(img1,img2,H,sz,st)

%% Image Resampling (Question 4b)

  % append a sufficient number of black columns to the left image  
  img1_ext = [img1, zeros(size(img1,1),sz)];
    
  m=repmat(1:size(img2, 2),[size(img2, 1),1]);
  m = m(:);
  XY1 = cart2hom ([repmat(1:size(img2, 1),[1,size(img2, 2)]), m(:)]');
  P = testH * XY1;
  P(1, :) = P(1, :) ./ P(3, :);
  P(2, :) = P(2, :) ./ P(3, :);
  
  % loop over all newly appended pixels plus some overlap    
  for y = 1:size(img2, 1)
      for x = 1:size(img, 2)
      
    
    % transform the current pixel coordinates to a point in the right image    
         p = H*[x;y;1];
    
    % look up gray-values of the four pixels nearest to the transformed
    % coordinates    
    % ...
    
    % bilinearly interpolate the gray-value at transformed coordinates and 
    % assign them to the source pixel in the left image. 
    % (Tip: use interpolate_2d.m for bilinear interpolation).
    % ...
      end
  end
    
    %% Intensity Adjustment (Question 4c)
    
    % linear weighting according to distance in horizontal direction
    % ...
  
    
  % end loop