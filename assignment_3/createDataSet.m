function [X, Y] = createDataSet(number)

% Create arbitrary datasets for evaluations
%
% INPUT:
% number                 : specify the number of points you want to create for each class
% xmin, xmax, ymin, ymax : range for the dataset
%
% OUTPUT:
% X			 : dataset
% Y			 : labeling
%
% INSTRUCTIONS:
% Use left mouse button to create data points. 
% You will get a message if the 'number' of points of the first class are created.
% As next step just press enter and continue with creating the data point of the second class

axis([-2, 2, -2, 2]); hold on;

x = ones(1,number);
y = ones(1,number);
for i = 1:number
	[x(i), y(i)] = ginput(1);
	plot(x(i), y(i), 'b*');
end

c1 = [x;y];

display('The data set of the first class are done');
display('Press enter and continue with creating data of the second class');
pause;

for i = 1:number
	[x(i), y(i)] = ginput(1);
	plot(x(i), y(i), 'r+');
end

c2 = [x;y];

X = [c1';c2']';
Y = [-ones(number,1);ones(number,1)]';
end