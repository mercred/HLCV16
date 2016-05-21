function vis_descriptor(PARAMS, CELLS)

%maxh is used to normalize color
maxh=0;
for by = 0:PARAMS.num_cells_height - 1
    for bx = 0:PARAMS.num_cells_width - 1
        maxh=max([maxh;CELLS{by+1, bx+1}]);
    end
end

i = 1;
for by = 0:PARAMS.num_cells_height - 1
    for bx = 0:PARAMS.num_cells_width - 1
      subplot(PARAMS.num_cells_height, PARAMS.num_cells_width, i);
      cell_hist_vis(PARAMS, CELLS{by+1, bx+1}, maxh);
      i = i + 1;
    end
end

function cell_hist_vis(PARAMS, h, maxh)
  h = h / maxh;

  r = floor(PARAMS.cellsize/2);
  for hidx = 0:length(h) - 1
    bin_center = PARAMS.hist_min + (hidx + 0.5)*PARAMS.hist_binsize;

    rx = cos(bin_center) * r;
    ry = sin(bin_center) * r;
    hval = h(hidx + 1);

    plot([-rx, rx], [-ry, ry], 'b-', 'LineWidth', 3, 'Color', [hval, hval, hval]);

    hold on;
  end

  set(gca, 'Color', [0, 0, 0]);
  set(gca, 'XTick', []);
  set(gca, 'YTick', []);
  axis ij;
  hold off;


