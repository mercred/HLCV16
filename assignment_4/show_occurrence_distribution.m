function show_occurrence_distribution(cluster_occurrences, cluster_idx)
    idx = 5;
    
    figure(2); hold on;
    
    list = randi([1,200],1,4);
    plot(0, 0, 'o', 'Color', 'g','LineWidth',1);
    plot(cluster_occurrences{list(4)}(2, :), cluster_occurrences{list(4)}(1, :), 'x', 'Color', 'y');
    plot(cluster_occurrences{list(1)}(2, :), cluster_occurrences{list(1)}(1, :), 'x', 'Color', 'k');
    plot(cluster_occurrences{list(2)}(2, :), cluster_occurrences{list(2)}(1, :), 'x', 'Color', 'r');
    plot(cluster_occurrences{list(3)}(2, :), cluster_occurrences{list(3)}(1, :), 'x', 'Color', 'b');
    
        
  % ...