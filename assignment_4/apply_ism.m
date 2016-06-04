function [detections, acc] = apply_ism(imgname, cluster_centers, cluster_occurrences)

 
  PARAMS = get_ism_params();

  % load the image
    image = rgb2gray(imread(imgname));
    img = single(image);
    
    %{
    figure(3)
    subplot(2, 1, 1);
    imshow(image); hold on;
    %}
    
    [px, py, H]=hessian(double(img), PARAMS.hessian_sigma, PARAMS.hessian_thresh);
    positions = [py';px'];
    votes = zeros(floor(size(img)./10) + 1);
    
    sift_frames = [px'; py'; PARAMS.feature_scale*ones(1, length(py)); PARAMS.feature_ori*ones(1, length(py))];
    [sift_frames, features] = vl_sift(single(img), 'Frames', sift_frames);
    features = single(features);
     
    for k = 1:size(features,2)
        distance = bsxfun(@minus, cluster_centers, features(:,k)).^2;
        distance = sqrt(sum(distance));
        indexes = find(distance < PARAMS.match_reco_tresh);
        
        %over activited codebook enties
        for n = 1:length(indexes)
            %over occurrences in codebook entry
            for m = 1:size(cluster_occurrences{indexes(n)}, 2)
                position = round(positions(:,k) + cluster_occurrences{indexes(n)}(:,m));
                if position(1) > 0 && position(2) > 0 && position(1) < size(img, 1) && position(2) < size(img, 2)
                    position = floor(position ./ 10) + 1;
                    votes(position(1), position(2)) = votes(position(1), position(2)) + 1.0 / size(cluster_occurrences{indexes(n)}, 2) / length(indexes);
                end
            end
        end
    end
    
    votes=nonmaxsup2d(votes);
    [vals, sortingIndexes] = sort(votes(:));
    sortingIndexes = flip(sortingIndexes);
    acc = flip(vals);
    
    [x,y] = getCordinates(sortingIndexes, size(votes, 1));
    x = x .* 10 - 5;
    y = y .* 10 - 5;
    
    detections = [y';x'];

    %{
    [x1,y1] = getCordinates(sortingIndexes(1), size(votes, 1))
    x1 = x1 * 10 - 5;
    y1 = y1 * 10 - 5;
    
    plot(x1,y1, 'x', 'Color', 'r'); hold on;
    subplot(2, 1, 2);
    imshow(votes);
    %}
    
    function [x,y] = getCordinates(index, dim_y)
        x = floor(index ./ dim_y);
        y = index - x .* dim_y;
        x = x + 1;
  % ...
 