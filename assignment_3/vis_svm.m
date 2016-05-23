%
% Visualization of the training data, support vectors, and linear SVM decision boundary.
% 
% The linear SVM decision boundary is defined by f(x) = w'*x + w0
%
% parameters:
%
% X - matrix of training points (one row per point), you can assume that points are 2 dimensional
% y - vector of point labels (+1 or -1)
% model.w and model.w0 are parameters of the decision function
% model.alpha is a vector of Lagrange multipliers

function vis_svm(X, y, model)

  % visualize positive and negative points   
  gscatter(X(:,1), X(:,2), y, 'br');  
  
  % visualize support vectors
  % 1. Get values and corresponding positions of alpha in descending order
  [val, ind] = sort(model.alpha, 1, 'descend');
  
  % 2. Limit up to a number of support vectors
  ind = ind(1:model.nsv);
  val = val(ind);
  
  % 3. Get (x,y) coordinates of support vectors
  X_sv = X(ind, :);  
  
  % 4. Plot data points with red circles around
  hold on;
  plot(X_sv(:,1), X_sv(:,2), 'ro', 'MarkerSize', 10);    

  % visualze decision boundary      
   x= linspace(-3, 3, 20);
   y= -model.w(1,1)*x/ model.w(2,1);
   plot(x,y);
    
  % end of visualize boundary
  
  min_x1 = min(X(:, 1));
  max_x1 = max(X(:, 1));
  min_x2 = min(X(:, 2));
  max_x2 = max(X(:, 2));             
  
  axis equal;
  axis([1.5*min_x1, 1.5*max_x1, 1.5*min_x2, 1.5*max_x2]);
  hold off;
end

