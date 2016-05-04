function [imgDx,imgDy]=gaussderiv(img,sigma)
    [d, x] = gaussdx(sigma);
    imgDx = conv2(img, d, 'same');
    imgDy = conv2(img, d', 'same');
end
