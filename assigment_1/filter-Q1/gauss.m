function [G, x] = gauss(sigma)
x = -sigma*3:1:sigma*3;
G = 1/sqrt(2*pi)/sigma*exp(-x.^2/(2*sigma^2));
end
% ... 
