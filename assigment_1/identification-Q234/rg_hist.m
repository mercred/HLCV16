%
%  compute joint histogram for r/g values
%  note that r/g values should be in the range [0, 1];
%  histogram should be normalized so that sum of all values equals 1
%
%  img_color - input color image
%  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
%

function h = rg_hist(img_color, num_bins)

  assert(size(img_color, 3) == 3, 'image dimension mismatch');
  assert(isfloat(img_color), 'incorrect image type');
  
  %define a 2D histogram  with "num_bins^2" number of entries
  h=zeros(num_bins + 1, num_bins + 1);

  r = img_color(:,:,1);
  g = img_color(:,:,2);
  b = img_color(:,:,3);
  
  r2 = r ./ (r+g+b);
  g2 = g ./ (r+g+b);
  
  r2 = floor(r2 ./ (1 / num_bins));
  g2 = floor(g2 ./ (1 / num_bins));
  
  one = ones(size(r2));
  r2 = r2 + one;
  g2 = g2 + one;
  
  for i=1:size(img_color,1)
    for j=1:size(img_color,2)
    
      h(r2(i,j), g2(i,j)) = h(r2(i,j), g2(i,j)) + 1;
    end
  end
  
  h = h ./ (size(r, 1) * size(r, 2));
  h = h(:);

  % ...

