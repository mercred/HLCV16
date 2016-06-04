function draw_detections(imgname, detections)
    
    image = rgb2gray(imread(imgname));
    
    box_x_size  = 110;
    box_y_size = 35;
    
    figure(4)
    imshow(image); hold on;
    bounding_box_x = [detections(2,1) - box_x_size, detections(2,1) + box_x_size, detections(2,1) + box_x_size, detections(2,1) - box_x_size, detections(2,1) - box_x_size];
    bounding_box_y = [detections(1,1) - box_y_size, detections(1,1) - box_y_size, detections(1,1) + box_y_size, detections(1,1) + box_y_size, detections(1,1) - box_y_size];
    
    plot(bounding_box_x, bounding_box_y, 'Color', 'g'); hold on;
    plot(detections(2,1), detections(1,1), 'x', 'Color', 'r'); hold on;
  % ... 


