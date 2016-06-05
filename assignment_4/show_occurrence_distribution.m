function show_occurrence_distribution(cluster_occurrences, cluster_idx)
    
    figure(2); hold on;
        
    image_centre = [50; 125]; 
    list = randi([1,200],1,4);
    plot(image_centre(2), image_centre(1), 'o', 'Color', 'g','LineWidth',1);
   
    show(cluster_occurrences{list(4)}, 'y', image_centre);
    show(cluster_occurrences{list(1)}, 'k', image_centre);
    show(cluster_occurrences{list(2)}, 'r', image_centre);
    show(cluster_occurrences{list(3)}, 'b', image_centre);
    %plot(cluster_occurrences{list(4)}(2, :), cluster_occurrences{list(4)}(1, :), 'x', 'Color', 'y');
    %plot(cluster_occurrences{list(1)}(2, :), cluster_occurrences{list(1)}(1, :), 'x', 'Color', 'k');
    %plot(cluster_occurrences{list(2)}(2, :), cluster_occurrences{list(2)}(1, :), 'x', 'Color', 'r');
    %plot(cluster_occurrences{list(3)}(2, :), cluster_occurrences{list(3)}(1, :), 'x', 'Color', 'b');
    
        
  % ...
  
function show(coordinates, color, image_centr)
  coordinates = -coordinates;
  coordinates = bsxfun(@plus, coordinates, image_centr);
  coordinates = [-coordinates(1,:) + image_centr(1)*2; coordinates(2,:)];

  plot(coordinates(2, :), coordinates(1, :), 'x', 'Color', color);