function [D, x] = gaussdx(sigma)

  x = [floor(-3.0*sigma + 0.5):floor(3.0*sigma + 0.5)];
  D = -(x.*exp(-x.^2/(2*sigma^2)))/(sqrt(2*pi)*sigma^3);
