from matplotlib import pyplot

def rectified(x):
    return max(0.0, x)

series_in = [x for x in range(-10, 11)]
series_out = [rectified(x) for x in series_in]

pyplot.plot(series_in, series_out)
pyplot.show()