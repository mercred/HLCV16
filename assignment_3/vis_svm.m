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
  hold on;
  for n = 1:length(X)
    if y(n)==1        
        plot(X(n,1),X(n,2),'k+','Color','r');
    else       
        plot(X(n,1),X(n,2),'k*','Color','b');
    end
  end
  
  % visualize support vectors
  % ...
  sup_vectors=find(model.alpha>0.1e-3);
  for n = 1:length(sup_vectors)
      plot(X(sup_vectors(n),1),X(sup_vectors(n),2),'ko','Color','m');
  end
      
  % visualze decision boundary 
  % ...  
  
  min_x1 = min(X(:, 1));
  max_x1 = max(X(:, 1));
  min_x2 = min(X(:, 2));
  max_x2 = max(X(:, 2)); 
  x1=[min_x1,max_x1];
  x2=-1.*(x1.*model.w(1)+model.w0)./model.w(2);
  plot(x1,x2);
  axis equal;
  axis([1.5*min_x1, 1.5*max_x1, 1.5*min_x2, 1.5*max_x2]);
  hold off;

