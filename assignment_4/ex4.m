show_q1 = true;
show_q2 = true;
show_q3 = true;

% order of addpath is important
addpath('./vlfeat-0.9.9/toolbox/kmeans');
addpath('./vlfeat-0.9.9/toolbox/sift');
addpath(['./vlfeat-0.9.9/toolbox/mex/' mexext]);

%
% Question 1: codebook generation
%

if show_q1
  num_clusters = 200;

  [cluster_centers, assignments, images]  = create_codebook('./cars-training', num_clusters);
 
  cluster_idx = 1;
  show_cluster_patches(images, assignments, cluster_idx);
end

%
% Question 2: occurrence generation
%

if show_q2
  cluster_occurrences = create_occurrences('./cars-training', cluster_centers);
  
  show_occurrence_distribution(cluster_occurrences, cluster_idx);
end

%
% Question 3: Recognition
%

if show_q3
    for i = 1:10
       imgname = sprintf('./cars-test/test-%d.png',i);
       %imgname = './cars-test/test-1.png';
       [detections, acc] = apply_ism(imgname, cluster_centers, cluster_occurrences);
       draw_detections(imgname, detections, 3+i, acc);
    end
end