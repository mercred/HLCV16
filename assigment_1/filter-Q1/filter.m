%% gauss.m (Question 1.a)

sigma = 4.0;
[gx, x] = gauss(sigma);

figure(1);
plot(x, gx, '.-');

%% gaussianfilter.m (Question 1.b)
 
img = double(rgb2gray(imread('graf.png')));
smooth_img = gaussianfilter(img, sigma);
figure(2);
subplot(1, 2, 1);
imagesc(img);
subplot(1, 2, 2);
imagesc(smooth_img);
colormap gray;

%% gaussdx.m (Question 1.c)

img = zeros(25, 25);
img(13, 13) = 1.0;
figure(3), imagesc(img)
colormap gray;

sigma = 6.0;
[G, x] = gauss(sigma);
[D, x] = gaussdx(sigma);

figure(4);
clf;
plot(x, G, 'b.-');
hold on;
plot(x, D, 'r-');


figure(5);
subplot(2, 3, 1);
imagesc(conv2(conv2(img, G, 'same'), G', 'same'));

subplot(2, 3, 2);
imagesc(conv2(conv2(img, G, 'same'), D', 'same'));

subplot(2, 3, 3);
imagesc(conv2(conv2(img, D', 'same'), G, 'same'));

subplot(2, 3, 4);
imagesc(conv2(conv2(img, D, 'same'), D', 'same'));

subplot(2, 3, 5);
imagesc(conv2(conv2(img, D, 'same'), G', 'same'));

subplot(2, 3, 6);
imagesc(conv2(conv2(img, G', 'same'), D, 'same'));

colormap gray;

%% gaussderiv.m (Question 1.d)

img_c = imread('graf.png');
img = double(rgb2gray(img_c));
[imgDx, imgDy] = gaussderiv(img, 6.0);

figure(6);
subplot(1, 3, 1);
imagesc(imgDx);

subplot(1, 3, 2);
imagesc(imgDy);

subplot(1, 3, 3);
imgmag = sqrt(imgDx.^2 + imgDy.^2);
imagesc(imgmag);
colormap gray;

...
