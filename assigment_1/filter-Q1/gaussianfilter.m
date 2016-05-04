function outimage=gaussianfilter(img,sigma)
    %{
    x = -sigma*6:1.0:sigma*6;
    [I, J] = ndgrid(x, x);
    I = I.^2;
    J = J.^2;
    F = I+J;
    F = 1/((2*pi)*sigma^2)*exp(-F/(2*sigma^2));
    outimage = conv2(img, F, 'same');
    %}
    [F, x] = gauss(sigma);
    outimage = conv2(img, F, 'same');
    outimage = conv2(outimage, F', 'same');
end
