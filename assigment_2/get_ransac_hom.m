%
% estimates the homography between two images by using potential 
% matching points between the images. Ransac is used for computation of
% the homography.
% 
%
% x1   : the x coordinates of the potential matching points in image 1
% y1   : the y coordinates of the potential matching points in image 1
% x2   : the x coordinates of the potential matching points in image 2
% y2   : the y coordinates of the potential matching points in image 2
% img1 : image 1
% img2 : image 2
%
% H    : Best estimated homography relating both input images using ransac
% 
function H = get_ransac_hom(x1,y1,x2,y2,img1,img2)
   
   % example: assume probability of true match : pInlier
   % -> all 4 sample pairs match: pInlier^4
   % find the amount of times k we have to draw in order to find a sample 
   % without outliers.
   % -> (1-pInlier^4)^k <= pFail (fail probability smaller than pFail)
   % -> log (pFail) / log (1-pInlier^4) <= k

   sample_size = size(x1,1); % we have sample_size samples
   pFail = 0.001; % we want to be 99.9 percent sure
   pInlier = 4 / sample_size; % just a very conservative guess for the amount of inlier (only 4);

   startTests = ceil (log(pFail) / log(1- pInlier^4));
   toDoTests = startTests; % adaptive maximum tests needed

   fprintf('number of points: %d\n\n', sample_size);
   fprintf('proportion of inliers: %f, number of RANSAC interations: %d\n', pInlier, toDoTests);

   pBestInlier = pInlier; % maximal fraction of inlier found so far
   H = eye(3);

   distInlier = 2; % the distance in pixel, distances below define inlier
   
   XY1 = cart2hom ([x1 , y1]');

   for j=1:startTests

     % draw sample and estimate homography 
     pos = uint32 (rand(4,1)' .* [sample_size-1, sample_size-1, sample_size-1, sample_size-1] + [1,1,1,1]);
     testH = get_hom( x1(pos), y1(pos), x2(pos), y2(pos) );

      % apply the homogrphy,
      P = testH * XY1;
      P(1, :) = P(1, :) ./ P(3, :);
      P(2, :) = P(2, :) ./ P(3, :);
      
      % count the inliers
      diff = P(1:2,:) - [x2, y2]';
      Norms = diff(1,:) .* diff(1,:) + diff(2,:) .* diff(2,:);
      inlierCount = length ( Norms(Norms < distInlier*distInlier));

      % adjust maximal iterations to be made
      % you can reestimate the probability for picking a true pair by
      % looking at the percentage of inliers of the current estimation
      pCurrentInlier = inlierCount / sample_size;

      if ( pCurrentInlier > pBestInlier)

        % find the amount of times k we have to draw a sample without outliers
        kTests = log(pFail) / log(1 - pCurrentInlier^4);
        toDoTests = round(kTests);

        fprintf('proportion of inliers: %f, number of RANSAC interations: %d\n', pCurrentInlier, toDoTests);

        % best Homography and amount of inliers
        H = testH;
        pBestInlier = pCurrentInlier;
      end

      if (toDoTests < j ) 
          break; % end here, we are pFail percent sure to have found H
      end
      
   end

   % probability of picking an inlier from the parameter set
   pInlier = pBestInlier; 
     
   % first find the inliers
   hitId = zeros(sample_size,1);
   for i = 1:sample_size
     p = H * [x1(i), y1(i), 1.0]';
     p = p / p(3);
     if (norm (p(1:2) - [x2(i),y2(i)]') < distInlier)
       hitId(i) = i;
     end
   end
   hitId = find (hitId);

   % re-estimate Homography
   H = get_hom( x1(hitId), y1(hitId), x2(hitId), y2(hitId) );

   disp('estimated H:/n');
   disp(H);

   hitId   = [];
   hitDist = [];
   for i = 1:sample_size
     p = H * [x1(i), y1(i), 1.0]';
     p = p / p(3);
     dist = norm (p(1:2) - [x2(i),y2(i)]');
     if ( dist < distInlier)
       hitId(end+1)   = i;
       hitDist(end+1) = dist;
     end
   end
   
   % visualization of inliners (use displaymatches)
   % use the distance of the matching pairs as sorting criteria
   figure(10);
   clf;

   displaymatches(img1, x1(hitId),y1(hitId), img2, x2(hitId), y2(hitId),...
     [1:size(hitId,2)], hitDist, min(20, size(x1, 1)));
   