%
% computes the coordinates of potential matching points between two images
% 
% img1 : image 1
% img2 : image 2
%
% px1 the x coordinates of the potential matching points in image 1
% py1 the y coordinates of the potential matching points in image 1
% px2 the x coordinates of the potential matching points in image 2
% py2 the y coordinates of the potential matching points in image 2
%
function [px1, py1, px2, py2] = get_corresponding_points(img1, img2)
    
   %  parameters, tested and working for the example images

   % Harris detector 
   sigma = 2;
   threshold = 3;

   % dxdy descriptor
   feature_window_size = 50;
   num_bins = 8;
   distanceThreshold = 0.025;

   % a) detection and description
   % Harris detection

   [px1 py1] = harris( img1, sigma, threshold );
   [px2 py2] = harris( img2, sigma, threshold );
   
   % discarding points too close to the border
   % consider feature_window_size for determining how much to discard

   imgBorderX = feature_window_size/2;
   imgBorderY = feature_window_size/2;
   [sizeY, sizeX] = size( img1 );
   % keep everything with a distance larger than imgBorderX, imgBorderY
   [px1, py1] = cutBorder(sizeX, sizeY, px1, py1, imgBorderX, imgBorderY);
   [sizeY, sizeX] = size( img2 );
   % keep everything with a distance larger than imgBorderX, imgBorderY
   [px2, py2] = cutBorder(sizeX, sizeY, px2, py2, imgBorderX, imgBorderY);

   % computing descriptors

   D1 = compute_descriptors(@dxdy_hist, img1, px1, py1, feature_window_size, num_bins);
   D2 = compute_descriptors(@dxdy_hist, img2, px2, py2, feature_window_size, num_bins);


  % b) distance computation

   D = get_point_dist(D1, D2);

   % c) best matches, fill function bestMatches
   [id1, id2, matchedScores] = match_points(D, distanceThreshold, 1);

   px1 = px1(id1);
   px2 = px2(id2);   
   py1 = py1(id1);   
   py2 = py2(id2);   

   figure(8);
   displaymatches(img1, px1, py1, img2, px2, py2, [1:size(px1,1)], ...
     matchedScores, min(20, size(px1,1)));


   

