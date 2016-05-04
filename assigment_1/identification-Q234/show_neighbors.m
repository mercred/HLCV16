%
% for each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
%
% note: use the previously implemented function 'find_best_match.m'
% note: use subplot command to show all the images in the same Matlab figure, one row per query image
%

function show_neighbors(model_images, query_images, dist_type, hist_type, num_bins)
  
  figure(4);
  clf;

  num_nearest = 5;
  index = 1;
  
  [best_match, D] = find_best_match(model_images, query_images, ...
                                dist_type, hist_type, num_bins);
  [vals, indexes] = sort(D);
  
  query_number = length(query_images);
  for i = 1:query_number
  
    %print query image
    print_image(query_images{i}, query_number, num_nearest + 1, index);
    index = index + 1;
    %print nearest images
    for j = 1:num_nearest
      print_image(model_images{indexes(j, i)}, query_number, num_nearest + 1, index);
      index = index + 1;
    end
  end
  % ...
function print_image(file_name, row_no, column_no, index)
    subplot(row_no, column_no, index); 
    imagesc(imread(file_name)); 
    set(gca,'xtick',[],'ytick',[])
   