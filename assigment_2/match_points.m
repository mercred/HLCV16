%
% computes the best matching point pairs between two images using a score
% value. Restricts the amount of matches of each point.
% 
% D         : Distance matrix, one entry for all possible pairs
% thresh    : the threshold for possible valid matches only pairs with a
%             score below that value are considered as possible pairs
% nMatches  : the maximal amount of matches for each point. No point is
%             allowed to occur in more than that value in the found pairs.
%
% id1 : array of indices of the matching pairs in image 1, a line index in D
% id2 : array of indices of the matching pairs in image 2, a column index in D
%
% matchedScores: the distance of the returned matches
%
function [id1, id2, matchedScores] = match_points(D, thresh, nMatches)

[idTemp1, idTemp2] = find (D < thresh);
matchScores = D( D < thresh );
Hits1 = zeros(size(D,1),1);
Hits2 = zeros(size(D,2),1);

% sort scores
[ sortSc sortId ] = sort(matchScores);

id1 = [];
id2 = [];
matchedScores = [];

% loop over sorted scores
for i = 1:size(matchScores)

  % get the indices of the potential pair
  ida = idTemp1(sortId(i));
  idb = idTemp2(sortId(i));

  % both indices are not already used more than nMatches
  if ~(Hits1(ida) > nMatches || Hits2(idb) > nMatches )
    % add the pair
    id1(end+1) = ida;
    id2(end+1) = idb;
    matchedScores(end+1) = D(ida, idb);

    % increase hit count for points of pair
    Hits1(ida) = Hits1(ida)+1;
    Hits2(idb) = Hits2(idb)+1;
  end
  
end

