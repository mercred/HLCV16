%
% model_images - list of file names of model images
% query_images - list of file names of query images
%
% dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
% hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
%
% note: use functions 'get_dist_by_name.m', 'get_hist_by_name.m' and 'is_grayvalue_hist.m' to obtain 
%       handles to distance and histogram functions, and to find out whether histogram function 
%       expects grayvalue or color image
%

function [best_match, D] = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)

  dist_func = get_dist_by_name(dist_type);
  hist_func = get_hist_by_name(hist_type);
  hist_isgray = is_grayvalue_hist(hist_type);

  D = zeros(length(model_images), length(query_images));
  
  model_hist = compute_histograms(model_images, hist_func, hist_isgray, num_bins);
  query_hist = compute_histograms(query_images, hist_func, hist_isgray, num_bins);

  % compute distance matrix

  for img_query = 1:length(query_images)
      for img_model = 1:length(model_images) 
        D(img_model, img_query) = dist_func(query_hist(img_query, :), model_hist(img_model, :));
      end
      fprintf('image "%s"\n', query_images{img_query});
  end
  
  [vals ,best_match] = min(D);



function image_hist = compute_histograms(image_list, hist_func, hist_isgray, num_bins)
  
  assert(iscell(image_list));
  image_hist = [];

  for i = 1:length(image_list)
    img2_color = imread(image_list{i});

    if hist_isgray
        img2_gray = double(rgb2gray(img2_color));
        image_hist(i, :) = hist_func(img2_gray, num_bins);
    else
        img2_color = double(img2_color);
        image_hist(i, :) = hist_func(img2_color, num_bins);
    end
  end

