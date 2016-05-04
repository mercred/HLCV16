% 
% compute and plot the recall/precision curve
%
% D - square matrix, D(i, j) = distance between model image i, and query image j
%
% note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image
%
function plot_rpc(D, plot_color)

  recall = [];
  precision = [];
  total_imgs = size(D, 2);

  num_images = size(D, 1);
  assert(size(D, 1) == size(D, 2));

  labels = eye(num_images);
  
  d = D(:);
  l = labels(:);
 
  [d, sortidx] = sort(d, 'ascend');
  l = l(sortidx);

  tp = 0;
  fp = 0;
  fn = sum(l);
  for idx = 1:length(d)
    tp = tp + l(idx);
    fp = fp + (l(idx)==0);
    
    % compute precision and recall values and append them to "recall" and "precision" vectors
    precision(idx) = tp/(tp+fp);
    recall(idx) = tp/fn;

  end

  plot(1-precision, recall, [plot_color '-'], 'LineWidth', 2);
