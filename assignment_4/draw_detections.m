function draw_detections(imgname, detections, window_id, acc)
    
    image = rgb2gray(imread(imgname));
    
    box_x_size  = 110;
    box_y_size = 35;
    figure(window_id)
    imshow(image); hold on;
    
    threshold = 1.5;
    i = 1;
    
    while acc(i) > threshold || i == 1     
        bounding_box_x = [detections(2,i) - box_x_size, detections(2,i) + box_x_size, detections(2,i) + box_x_size, detections(2,i) - box_x_size, detections(2,i) - box_x_size];
        bounding_box_y = [detections(1,i) - box_y_size, detections(1,i) - box_y_size, detections(1,i) + box_y_size, detections(1,i) + box_y_size, detections(1,i) - box_y_size];
        
        if i > 1
            if if_overlap(detections(:,i), detections(:,1:i-1), 1.5* box_x_size)
                i = i + 1;
                continue;
            end
        end
        
        plot(bounding_box_x, bounding_box_y, 'Color', 'g'); hold on;
        plot(detections(2,i), detections(1,i), 'x', 'Color', 'r'); hold on;
        
        i = i+1;
    end
  % ... 
function overlap = if_overlap(x, list, min_dist)

    overlap = false;
    for i = 1:size(list, 2)
        if norm(x - list(:,i)) < min_dist
            overlap = true;
        end
    end

