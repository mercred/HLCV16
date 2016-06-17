% --------------------------------------------------------------------
function [im, labels] = getBatch_1(imdb, batch)
%getBatch is called by cnn_train.

%'imdb' is the image database.
%'batch' is the indices of the images chosen for this batch.

%'im' is the height x width x channels x num_images stack of images. If
%opts.batchSize is 50 and image size is 64x64 and grayscale, im will be
%64x64x1x50.
%'labels' indicates the ground truth category of each image.

%This function is where you should 'jitter' data.

% Add jittering here before returning im
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

%%% Supplement Code
% Supplement code here! 
% --------------------------------------------------------------------
batchSize = size(batch, 2);
for i = 1:batchSize            
    if rand() > 0.5
        % Randomly flip image from left to right
        image = im(:,:,:,i);             
        flipped_img = flip(image,2);        
        im(:,:,:,i) = flipped_img;   
    end    
end

end
