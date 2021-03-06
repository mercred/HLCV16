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
  [height, width] = size(img1);
  img = zeros([height, width+sz]);
  img(1:height, 1:width) = img1;
  
  % loop over all newly appended pixels plus some overlap    
  for x = width-st:width
    for y = 1:height                
        % transform the current pixel coordinates to a point in the right image            
        % homogeneous and Cartesian coordinates
        hom = H*[x;y;1];
        cart = [hom(1)/hom(3), hom(2)/hom(3)];
        
        % look up gray-values of the four pixels nearest to the transformed
        % coordinates                            
        % bilinearly interpolate the gray-value at transformed coordinates and 
        % assign them to the source pixel in the left image. 
        % (Tip: use interpolate_2d.m for bilinear interpolation).        
        gray_left = img(y,x);
        gray_right = interpolate_2d(img2, cart(2), cart(1));            
        
        %% Intensity Adjustment (Question 4c)    
        % linear weighting according to distance in horizontal direction
        img(y,x) = (gray_left*(width-x) + gray_right*(x-width+st))/st;
    % end loop
    end
  end
  
  % Copy part of the right image that does not overlap  
  for x = width+1:width+sz
      for y = 1:height
        hom = H*[x;y;1];
        cart = [hom(1)/hom(3), hom(2)/hom(3)];
        gray_value = interpolate_2d(img2, cart(2), cart(1));
        img(y,x) = gray_value;
      end
  end
end
