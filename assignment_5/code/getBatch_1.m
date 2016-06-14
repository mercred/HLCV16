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
im2 = imdb.images.data(:,:,:,batch) ;
labels2 = imdb.images.labels(1,batch) ;

%%% Supplement Code
% Supplement code here! 
% --------------------------------------------------------------------

im = zeros(size(im2, 1), size(im2, 2), size(im2, 3), size(im2, 4) * 2, 'single');
labels = zeros(1,length(labels2) * 2, 'single');
labels(1:length(labels2)) = labels2;
labels(length(labels2)+1:length(labels2)*2) = labels2;
im(:,:,:,1:size(im2, 4)) = im2;
x = size(im2, 4);
for i = 1:x
    im(:,:,1,x+i) = fliplr(im2(:,:,1,i));
end

%{
x = size(im, 4);
for i = 1:x
    im(:,:,1,i) = fliplr(im(:,:,1,i));
end
%}
end
