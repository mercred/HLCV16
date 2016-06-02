function show_cluster_patches(images, assignments, cluster_idx)
      
      figure(cluster_idx);
      axis off;
      num_clusters = 7;
      image_number_per_row = 7;
      
      for i = 1:randi([1,100],1,num_clusters)
        indexes = find(assignments == i);
        indexes = indexes(1:min(length(indexes), image_number_per_row));
        
        for k = 1:length(indexes)
            subplot(num_clusters, image_number_per_row, (i-1)*image_number_per_row+k);
            imshow(images{indexes(k)});
        end
        
      end
  end
% ...