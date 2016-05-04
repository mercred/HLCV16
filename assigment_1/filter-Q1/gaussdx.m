function [D, x] = gaussdx(sigma)
    x = -sigma*3:1:sigma*3;
    D = -1/sqrt(2*pi)/sigma^3*x.*exp(-x.^2/(2*sigma^2));
% ...