function I2 = imresize_simple(I, scale)

  is_uint8 = isinteger(I);

  I = double(I);

  if length(size(I)) == 2
    I2 = imresize_simple_helper(I, scale);
  else
    for idx = 1:size(I, 3)
      I2(:, :, idx) = imresize_simple_helper(squeeze(I(:, :, idx)), scale);
    end
  end

  if is_uint8
    I2 = uint8(I2);
  end


function I2 = imresize_simple_helper(I, scale)

  assert(isfloat(I));
  assert(length(size(I)) == 2);
  
  xi = linspace(1,size(I,2), scale*size(I,2));  %# interpolated horizontal positions
  yi = linspace(1,size(I,1), scale*size(I,1));  %# interpolated vertical positions

  [X Y] = meshgrid(1:size(I,2),1:size(I,1));   %# pixels X-/Y-coords
  [XI YI] = meshgrid(xi,yi);                   %# interpolated pixels X-/Y-coords

  I2 = interp2(X,Y,I, XI,YI, '*linear');        %# interp values at these positions
