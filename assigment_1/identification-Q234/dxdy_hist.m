%
%  compute joint histogram of Gaussian partial derivatives of the image in x and y direction
%  for sigma = 6.0, the range of derivatives is approximately [-34, 34]
%  histogram should be normalized so that sum of all values equals 1
%
%  img_gray - input grayvalue image
%  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
%
%  note: you can use the function gaussderiv.m from the filter exercise.
%

function h=dxdy_hist(img_gray, num_bins)

  assert(length(size(img_gray)) == 2, 'image dimension mismatch');
  assert(isfloat(img_gray), 'incorrect image type');

  % compute the first derivatives
  sigma = 6.0;
  [imgDx,imgDy]=gaussderiv(img_gray,sigma);

  % quantize derivatives to "num_bins" number of values
  one = ones(size(imgDx));
  imgDx = imgDx + one .* 34;
  imgDx = floor(imgDx ./ (68 / num_bins));
  
  imgDy = imgDy + one .* 34;
  imgDy = floor(imgDy ./ (68 / num_bins));

  % define a 2D histogram  with "num_bins^2" number of entries
  h=zeros(num_bins,num_bins);
  
  for i=1:size(img_gray,1)
    for j=1:size(img_gray,2)

      %increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
      h(imgDx(i,j), imgDy(i,j)) = h(imgDx(i,j), imgDy(i,j)) + 1;

    end
  end
  h = h ./ (size(img_gray,1) * size(img_gray,2));
  h = h(:);
end
