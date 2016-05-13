function [pxc, pyc] = cutBorder(dx, dy, px, py, borderX, borderY)
% dx dy size of the image
% borderX and borderY defines which pixels are seen as border
% and have to be thrown away
% px and py are the indices in the image

pxc = px((px > borderX) & (py > borderY) & (px < dx - borderX) & (py < dy - borderY));
pyc = py((px > borderX) & (py > borderY) & (px < dx - borderX) & (py < dy - borderY));
