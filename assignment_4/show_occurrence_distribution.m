function show_occurrence_distribution(cluster_occurrences, cluster_idx)
    idx = 5;
    
    figure(2); hold on;
    
    list = randi([1,200],1,4);
    plot(cluster_occurrences{list(4)}(2, :), cluster_occurrences{list(4)}(1, :), 'x', 'Color', 'y'); hold on;
    plot(cluster_occurrences{list(1)}(2, :), cluster_occurrences{list(1)}(1, :), 'x', 'Color', 'r'); hold on;
    plot(cluster_occurrences{list(2)}(2, :), cluster_occurrences{list(2)}(1, :), 'x', 'Color', 'g'); hold on;
    plot(cluster_occurrences{list(3)}(2, :), cluster_occurrences{list(3)}(1, :), 'x', 'Color', 'b'); hold on;
    
        
  % ...