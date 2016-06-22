show_init_cnn = false;
show_q1 = false;
show_q2 = false;
show_q3 = false;
show_q4 = true;


%setting up Matconvnet framework (Can be commented after the first compilation)
%If your machine has GPU, you can speed up training CNNs by uncommenting
%the line: vl_compilenn('enableGpu', true)
% init_matconvnet()

%
% Initial Performance of Toy CNN
%

if show_init_cnn


% Prepare data

% The cnn_init function specifies the network architecture.
net = cnn_init();

% The setup_data function loads the training and testing images into
% MatConvNet's imdb structure. 

imdb = setup_data();

% set parameters of CNN such as batchsize, learning rate, number of epochs
opts=options();


%  Train & Test

[~,info]=cnn_train(net, imdb, @getBatch, opts, 'val', find(imdb.images.set == 2)) ;

fprintf('Lowest validation erorr for initial CNN is %f\n',min(info.val.error(1,:)))

end
%
% Question 1 (Data Augmentation)
%

if show_q1
    
% Prepare data

% The cnn_init function specifies the network architecture.
net = cnn_init();

% The setup_data function loads the training and testing images into
% MatConvNet's imdb structure. 

imdb = setup_data();

% set parameters of CNN such as batchsize, learning rate, number of epochs
opts=options();

%Supplement function getBatch_1 to do data augmentation

%  Train & Test

[~,info]=cnn_train(net, imdb, @getBatch_1, opts, 'val', find(imdb.images.set == 2)) ;

fprintf('Lowest validation erorr for data augmented CNN is %f\n',min(info.val.error(1,:)))


end

%
% Question 2 (Mean Subtraction)
%

if show_q2

% Prepare data

% The cnn_init function specifies the network architecture.
net = cnn_init();

% The setup_data function loads the training and testing images into
% MatConvNet's imdb structure. 

imdb = setup_data_2();

% set parameters of CNN such as batchsize, learning rate, number of epochs
opts=options();


%  Train & Test

[~,info]=cnn_train(net, imdb, @getBatch, opts, 'val', find(imdb.images.set == 2)) ;

fprintf('Lowest validation erorr for mean subtracted CNN is %f\n',min(info.val.error(1,:)))

    
end
%
% Question 3 (Drop out layer)
%

if show_q3
    % Prepare data

% The cnn_init function specifies the network architecture.
net = cnn_init_3();

% The setup_data function loads the training and testing images into
% MatConvNet's imdb structure. 

imdb = setup_data();

% set parameters of CNN such as batchsize, learning rate, number of epochs
opts=options();


%  Train & Test

[~,info]=cnn_train(net, imdb, @getBatch, opts, 'val', find(imdb.images.set == 2)) ;

fprintf('Lowest validation erorr for CNN with dropout is %f\n',min(info.val.error(1,:)))

    
end

%
% Question 2 (Deeper Toy CNN)
%

if show_q4
    
% Prepare data

% The cnn_init function specifies the network architecture.
net = cnn_init_4();

% The setup_data function loads the training and testing images into
% MatConvNet's imdb structure. 

imdb = setup_data_2();

% set parameters of CNN such as batchsize, learning rate, number of epochs
opts=options();


%  Train & Test

[~,info]=cnn_train(net, imdb, @getBatch_1, opts, 'val', find(imdb.images.set == 2)) ;

fprintf('Lowest validation error for Deeper CNN with previous parts changes is %f\n',min(info.val.error(1,:)))

    
end