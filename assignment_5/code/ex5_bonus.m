show_part_a = false;
show_part_b= true;


%setting up Matconvnet framework, downloading VGG_F and running a VGG-F demo(Can be commented after the first compilation)
%If your machine has GPU, you can speed up training CNNs 
%by uncommenting vl_compilenn('enableGpu', true) 
% quickStartDemo()

%
% Question 5 part a
%

if show_part_a
    
% load the pre-trained CNN
net = load('imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;

%% extract deCAF feature for each image
% Choose only 100 images for train and 100 images for test for each class
% make sure your training set and test set is the same as previous parts.
% Supplement Code Here

% Load data
total_images = 1500;
batch_size = 30;
imdb = setup_data();

% Divide data into training and test sets
train_data = imdb.images.data(:,:,:, imdb.images.set == 1);
test_data = imdb.images.data(:,:,:, imdb.images.set == 2);
train_labels = imdb.images.labels(:, imdb.images.set == 1);
test_labels = imdb.images.labels(:, imdb.images.set == 2);
label_list = unique(imdb.images.labels);

% Preprocessing: mean img subtraction
new_data = single(zeros(224,224,3,total_images*2));
for i = 1:total_images*2
    if i <= total_images
        img = train_data(:,:,:,i);
    else
        img = test_data(:,:,:,i-total_images);
    end
    img = single(img);
    img = imresize(img, net.meta.normalization.imageSize(1:2));
    img = img - rgb2gray(net.meta.normalization.averageImage);
    img = repmat(double(img)./255,[1 1 3]); % convert back to RGB
    new_data(:,:,:,i) = img;
end

% Obtain features for SVM by running VGG-F network
sets = {'train', 'test'};
features = zeros(total_images*2, 64*64);
index = 1;
start = 1; fin = batch_size;
for set = 1:length(sets)    
    for i = 1:total_images/batch_size
        % Get batch of data
        batch = new_data(:,:,:,start:fin);        
        
        % Run VGG-F network on batch of data and save the result as a feature
        res = vl_simplenn(net, batch);    
        res = res(19).x;
        features(start:fin,:) = squeeze(res(1,1,:,:))';
        
        % Update indices
        start = start + batch_size;
        fin = fin + batch_size;
    end
end

% Set deCAF structure
deCAF = struct('train_data', double(features(1:total_images,:)), 'train_labels', double(train_labels'), ...
               'test_data', double(features(total_images+1:end,:)), 'test_labels', double(test_labels'), ...
               'label_list', label_list' ...
              );


%% apply Linear SVM for classification
%deCAF variable contain deCAF features for each image and the corresponding
%lable. make sure your training set and test set is the same as previous
%parts. LinearSVMClassifier should be suplemented.
LinearSVMClassifier(deCAF)


end
%
% Question 5 part b
%
if show_part_b

bonus_options;

%configure the Net

net = bonus_cnn_init();

% Prepare data

imdb = bonus_setup_data(net.normalization.averageImage);


%Train

[net, info] = cnn_train(net, imdb, @getBatch_bonus, ...
    opts, ...
    'val', find(imdb.images.set == 2)) ;
fprintf('Lowest validation error for Deeper CNN with previous parts changes is %f\n',min(info.val.error(1,:)))

end