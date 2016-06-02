function [cluster_centers, assignments, subimages] = create_codebook(sDir, num_clusters)
  
  PARAMS = get_ism_params();

  vImgNames = dir(fullfile(sDir, '*.png'));
  features = cell(1,length(vImgNames));
  subimages = cell(0);

  for i = 1:length(vImgNames)
      img = imread([sDir, '/', vImgNames(i).name]);
      img = img(:,:,1);
      [px py, H]=hessian(double(img), PARAMS.hessian_sigma, PARAMS.hessian_thresh);
      
      subimages = [subimages, get_sub_image(px, py, img)];
      
      sift_frames = [px'; py'; ...
      PARAMS.feature_scale*ones(1, length(py)); ...
      PARAMS.feature_ori*ones(1, length(py))];
      [sift_frames, features{i}] = vl_sift(single(img), 'Frames', sift_frames);
      
  end
  points = [features{:}];
  [cluster_centers, assignments] = vl_kmeans(single(points), min(num_clusters, size(points,2)));
    
 function subimage = get_sub_image(px, py, image)
    subimage = cell(1,length(px));
    size_img = 16;
    lx = max(px - size_img, 1);
    rx = min(px + size_img, size(image, 2));
    uy = max(py - size_img, 1);
    ly = min(py + size_img, size(image, 1));
    for i=1:length(px) 
        subimage{i} = image(uy(i):ly(i), lx(i):rx(i));
    end


