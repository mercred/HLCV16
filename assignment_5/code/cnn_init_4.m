function net = cnn_init()
% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

% The cnn_init function specifies the network architecture. You will be
% modifying the function.

%code for Computer Vision, Georgia Tech by James Hays
%based of the MNIST example from MatConvNet

rng('default');
rng(0);

% constant scalar for the random initial network weights. You shouldn't
% need to modify this.
f=1/100; 

net.layers = {} ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,1,10, 'single'), zeros(1, 10, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'conv1') ;                                              
                           ... %% first layer
% Supplement Code Here
% reduce the pooling window and stride
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool',  [3,3], ...
                           'stride', 3, ...
                           'pad', 0) ;
%

net.layers{end+1} = struct('type', 'relu') ;

% Add new layers here 
% Supplement code here
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(3,3,1,10, 'single'), zeros(1, 10, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'conv2') ;                       
                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool',  [2,2], ...
                           'stride', 2, ...
                           'pad', 0) ;     
                       
net.layers{end+1} = struct('type', 'relu') ;                       

net.layers{end+1} = struct('type', 'dropout', ...                           
                           'rate', 0.5);
% Supplement Code Here                       
% added drop out layer from previous part


%the fc layer from original toy cnn.
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(9,9,10,15, 'single'), zeros(1, 15, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'fc1') ;
                      
% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% Visualize the network
vl_simplenn_display(net, 'inputSize', [64 64 1 50])
