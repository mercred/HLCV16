%
%  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
%  assume that image intensity varies between 0 and 255
%
%  img_gray - input image in grayscale format
%  num_bins - number of bins in the histogram
%
function h = normalized_hist(img_gray, num_bins)
  
  assert(length(size(img_gray)) == 2, 'image dimension mismatch');
  assert(isfloat(img_gray), 'incorrect image type');

  sum_img = 0;
  img_gray = floor(img_gray ./ (256 / num_bins));
  for i = 1:num_bins
    h(i) = sum(sum(img_gray == i-1));
    sum_img = sum_img + h(i);
  end
  
  for i = 1:num_bins
    h(i) = h(i) / sum_img;
  end
  h = h(:);
end