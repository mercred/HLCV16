function net = cnn_init_3()
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
                           'weights', {{f*randn(9,9,1,10, 'single'), zeros(1, 10, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'conv1') ;
                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [7 7], ...
                           'stride', 7, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'relu') ;

%Supplement Code Here
% Insert Dropout layer here.       
net.layers{end+1} = struct('type', 'dropout', ...                           
                           'rate', 0.25);


net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(8,8,10,15, 'single'), zeros(1, 15, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'fc1') ;
% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% Visualize the network
vl_simplenn_display(net, 'inputSize', [64 64 1 50])
