function show_cluster_patches(images, assignments, cluster_idx)
    figure(cluster_idx);
    axis off;      
    images_to_show = 30;
    images_in_a_row=7;
    indexes = find(assignments == cluster_idx,images_to_show);     
    for k = 1:length(indexes) 
          subplot(floor(images_to_show/images_in_a_row)+1, images_in_a_row, k);
          imshow(images{indexes(k)});
    end
end
% ...