function imdb = bonus_setup_data(averageImage)
%code for Computer Vision, Georgia Tech by James Hays

% It's probably easiest if you start by copy/paste the contents of
% ex5_setup_data, including the normalization by removing the mean.

% [copied from the project webpage]
% The input images need to be resized to 224x224. More specifically, the
% input images need to be 224x224 when returned by getBatch(). You could
% keep them at higher resolution in imdb and crop them to 224x224 as a form
% of jittering. See cnn_imagenet_tch.m for an extreme jittering
% example. You can call that function if you want, but you can achieve high
% accuracy with no jittering. You could also simply reuse your jittering
% strategy from main exercise.
   
% VGG-F accepts 3 channel (RGB) images. The 15 scene database contains
% grayscale images. There are two possibilities: modify the first layer of
% VGG-F to accept 1 channel images, or concatenate the grayscale images
% with themselves (e.g. cat(3, im, im, im)) to make an RGB image. The
% latter is probably easier and safer.
  
% VGG-F expects input images to be normalized by subtracting the average
% image, just like in the main exercise. VGG-F provides a 224x224x3 average image in
% net.normalization.averageImage.

% Supplement Code Here

%This path is assumed to contain 'test' and 'train' which each contain 15
%subdirectories. The train folder has 100 samples of each category and the
%test has an arbitrary amount of each category. This is the exact data and
%train/test split.
SceneJPGsPath = '../data/15SceneData/';

num_train_per_category = 100;
num_test_per_category  = 100; %can be up to 110
total_images = 15*num_train_per_category + 15 * num_test_per_category;

image_size = [224 224]; %downsampling data for speed and because it hurts
% accuracy surprisingly little

imdb.images.data   = zeros(image_size(1), image_size(2), 3, total_images, 'single');
imdb.images.labels = zeros(1, total_images, 'single');
imdb.images.set    = zeros(1, total_images, 'uint8');
image_counter = 1;

categories = {'bedroom', 'coast', 'forest', 'highway', ...
              'industrial', 'insidecity', 'kitchen', ...
              'livingroom', 'mountain', 'office', 'opencountry', ...
              'store', 'street', 'suburb', 'tallbuilding'};
          
sets = {'train', 'test'};

fprintf('Loading %d train and %d test images from each category\n', ...
          num_train_per_category, num_test_per_category)

%Read each image and resize it to 224x224
for set = 1:length(sets)
    for category = 1:length(categories)
        cur_path = fullfile( SceneJPGsPath, sets{set}, categories{category});
        cur_images = dir( fullfile( cur_path,  '*.jpg') );
        
        if(set == 1)
            fprintf('Taking %d out of %d images in %s\n', num_train_per_category, length(cur_images), cur_path);
            cur_images = cur_images(1:num_train_per_category);
        elseif(set == 2)
            fprintf('Taking %d out of %d images in %s\n', num_test_per_category, length(cur_images), cur_path);
            cur_images = cur_images(1:num_test_per_category);
        end

        for i = 1:length(cur_images)
            cur_image = imread(fullfile(cur_path, cur_images(i).name));            
            cur_image = single(cur_image);            
            cur_image = imresize(cur_image, image_size);                       
            
            % Stack images into a large 224 x 224 x 3 x total_images matrix
            % images.data            
            cur_image = cat(3, cur_image, cur_image, cur_image); % convert to RGB
            imdb.images.data(:,:,:,image_counter) = cur_image;            
            imdb.images.labels(  1,image_counter) = category;
            imdb.images.set(     1,image_counter) = set; %1 for train, 2 for test (val?)
            
            image_counter = image_counter + 1;
        end         
         %%%% Supplement code here to do mean subtraction.        
         if (set == 1)
             % Train set only
             inds = imdb.images.set == set;
             data = imdb.images.data(:,:,:, inds);             
             N = size(data,4);                          
             
             % Subtract mean image                          
             for i = 1:total_images
                 if inds(i) > 0
                     imdb.images.data(:,:,:,i) = imdb.images.data(:,:,:,i) - averageImage;
                 end                 
             end 
         end         
         %%%%%%%         
    end
end



