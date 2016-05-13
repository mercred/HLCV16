% input: p1 and p2 are 3xN matrices, each column is a point in homogeneous coordinates (i.e. [x; y; 1]
% output: matrix M, such that p2 = M*p1 for the case of input points without noise (otherwise M is a least squares solution).
%

function M = get_affine(p1, p2)

  % ... 
