function cluster_occurrences = create_occurrences(sDir, cluster_centers)
  
  PARAMS = get_ism_params();
  cluster_occurrences = cell(size(cluster_centers,2), 1);
  vImgNames = dir(fullfile(sDir,'*.png'));
  
  for i = 1:length(vImgNames)
    img = imread([sDir, '/', vImgNames(i).name]);
    img = img(:,:,1);
    [px, py, H]=hessian(double(img), PARAMS.hessian_sigma, PARAMS.hessian_thresh);
    positions = [py';px'];
    
    sift_frames = [px'; py'; PARAMS.feature_scale*ones(1, length(py)); PARAMS.feature_ori*ones(1, length(py))];
    [sift_frames, features] = vl_sift(single(img), 'Frames', sift_frames);
    features = single(features);
      
    for k = 1:size(features,2)
        distance = bsxfun(@minus, cluster_centers, features(:,k)).^2;
        distance = sqrt(sum(distance));
        indexes = find(distance < PARAMS.match_tresh);
        
        v = size(img)' ./ 2 - positions(:,k);
        for n = 1:length(indexes)
            %append(cluster_occurrences{indexes(n)}, v);
            cluster_occurrences{indexes(n)} = [cluster_occurrences{indexes(n)}, v];
        end
    end
  end

  % ...