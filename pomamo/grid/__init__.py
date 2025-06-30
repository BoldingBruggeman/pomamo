import numpy
import scipy.interpolate

points = numpy.array(((0., 0.1), (10., 1.), (25., 2.), (60., 4.), (100., 5.), (200., 10.), (500., 50.), (2000., 100.), (10000., 200.)), dtype=float)

def generate(depth, plot=False):
    if depth < 1.:
        h = numpy.empty((10,))
        h[:] = depth / 10
        return h

    x_sf, y_sf = points[:, 0], points[:, 1]
    grid = numpy.linspace(0., depth, 1000)
    res = scipy.interpolate.pchip_interpolate(x_sf, y_sf, grid)
    x_bt, y_bt = (depth - x_sf - y_sf)[::-1], y_sf[::-1]
    botres = scipy.interpolate.pchip_interpolate(x_bt, y_bt, grid)
    z_switch = numpy.interp(0., res - botres, grid)
    #print('Bottom resolution dominates from %.3f m' % z_switch)
    stop_sf = max(1, x_sf.searchsorted(z_switch - 0.15 * depth))
    start_bt = min(x_bt.searchsorted(z_switch + 0.15 * depth), x_bt.size - 1)
    x = numpy.concatenate((x_sf[:stop_sf], (z_switch - 0.1 * depth, z_switch + 0.1 * depth), x_bt[start_bt:]))
    y = numpy.concatenate((y_sf[:stop_sf], (numpy.interp(z_switch - 0.1 * depth, x_sf, y_sf), numpy.interp(z_switch + 0.1 * depth, x_bt, y_bt)), y_bt[start_bt:]))
    minres = scipy.interpolate.pchip_interpolate(x, y, grid)

    hs = []
    zs = []
    z = 0
    while z < depth:
        h = numpy.interp(z, grid, minres)
        zs.append(z + h/2)
        hs.append(h)
        z += h
    #print('%i points' % len(final_depths))

    if plot:
        from matplotlib import pyplot
        fig = pyplot.figure()
        ax = fig.gca()
        ax.plot(grid + res/2, res)
        ax.plot(grid + botres/2, botres)
        ax.plot(zs, hs, '.')
        pyplot.show()

    return numpy.array(hs)
